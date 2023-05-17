import torch
import pytorch_lightning as pl
from .cnn2d import compile_original_2D_CNN,compile_VGG_CNN
from torchmetrics import MeanSquaredError, F1Score, Precision, Recall, R2Score, Dice, JaccardIndex
from torchvision.models import resnet18
import segmentation_models_pytorch as smp

class ForestModel(pl.LightningModule):

    def __init__(self, input_width, lr, loss_fn_weight, architecture, dropout=0, weight_decay=0, task="pixel",
                 alpha=0, gamma=0, loss_fn="BCE"):
        super().__init__()
        self.save_hyperparameters()

        if architecture == "VGG":
            self.model = compile_VGG_CNN(input_width=input_width)
        elif architecture == "RESNET":
            self.model = resnet18(num_classes=1)
            self.model.conv1 = torch.nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False)
        elif task == "tile_segmentation":
            self.model = smp.Unet("resnet18", classes=1, in_channels=8, encoder_weights=None)
        else:
            self.model = compile_original_2D_CNN(input_width=input_width, dropout=dropout)

        self.task = task
        if self.task == "tile_regression":
            self.mse_metric = MeanSquaredError()
            self.rmse_metric = MeanSquaredError(squared=False)
            self.r2_metric = R2Score()
        else:
            # loss_fn_weight = None if loss_fn_weight == 0 else torch.tensor([loss_fn_weight])
            # self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=loss_fn_weight)
            self.f1_metric = F1Score(task="binary", threshold=0.5, ignore_index=-1)
            self.precision_metric = Precision(task="binary", threshold=0.5, ignore_index=-1)
            self.recall_metric = Recall(task="binary", threshold=0.5, ignore_index=-1)

        if self.task == "tile_segmentation":
            self.iou = JaccardIndex(task="binary", num_classes=1, threshold=0.5, ignore_index=-1)

        if loss_fn == "DiceLoss":
            self.loss_fn = smp.losses.DiceLoss(mode="binary", ignore_index=-1)
        elif loss_fn == "MSELoss":
            self.loss_fn = torch.nn.MSELoss()
        else:
            loss_fn_weight = None if loss_fn_weight == 0 else torch.tensor([loss_fn_weight])
            self.loss_fn = smp.losses.SoftBCEWithLogitsLoss(pos_weight=loss_fn_weight, ignore_index=-1)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fn_name = loss_fn

    def forward(self, features):
        output = self.model(features)

        if len(output.shape) == 2:
            output = output.squeeze()
        if len(output.shape) == 0:
            output = output.unsqueeze(dim=0)

        return output
    
    def shared_step(self, batch, stage):

        features = batch[0]
        target = batch[1]
        
        output = self.forward(features)
        if self.task == "tile_regression" and self.loss_fn_name == "MSELoss":
            output = torch.sigmoid(output)

        if self.task == "pixel" and self.loss_fn_name == "DiceLoss":
            loss = self.loss_fn(output.unsqueeze(1), target.unsqueeze(1))
        else:
            loss = self.loss_fn(output, target)

        if self.alpha != 0 and self.gamma != 0:
            loss_exp = torch.exp(-loss)
            loss = self.alpha * (1 - loss_exp) ** self.gamma * loss

        metrics_batch = {"loss": loss}

        if self.task == "tile_regression":
            if self.loss_fn_name != "MSELoss":
                output = torch.sigmoid(output)
            metrics_batch["mse"] = self.mse_metric(output, target)
            metrics_batch["rmse"] = self.rmse_metric(output, target)
            metrics_batch["r2"] = self.r2_metric(output, target)
        else:
            output = torch.sigmoid(output)
            metrics_batch["f1"] = self.f1_metric(output, target)
            metrics_batch["precision"] = self.precision_metric(output, target)
            metrics_batch["recall"] = self.recall_metric(output, target)

            if self.task == "tile_segmentation":
                metrics_batch["iou"] = self.iou(output, target.type(torch.int))

        if stage == "train":
            self.training_step_outputs.append(metrics_batch)
        if stage == "valid":
            self.validation_step_outputs.append(metrics_batch)
        if stage == "test":
            self.test_step_outputs.append(metrics_batch)

        return loss

    def shared_epoch_end(self, outputs, stage):

        metrics_epoch = {}
        for key, value in outputs[0].items():
            metrics_epoch[f"{stage}_{key}"] = torch.stack([x[key] for x in outputs]).mean()

        self.log_dict(metrics_epoch, prog_bar=True)
        return metrics_epoch

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        result = self.shared_epoch_end(outputs, "train")
        self.training_step_outputs.clear() 
        return result

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        result = self.shared_epoch_end(outputs, "valid")
        self.validation_step_outputs.clear() 
        return result

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        result = self.shared_epoch_end(outputs, "test")
        self.test_step_outputs.clear()
        return result
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        features = batch[0]
        target = batch[1]

        output = self.forward(features)
        output = torch.sigmoid(output)
        return {"preds": output, "target": target}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)