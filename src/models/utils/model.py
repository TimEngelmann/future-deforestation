import torch
import pytorch_lightning as pl
from .cnn2d import compile_original_2D_CNN,compile_VGG_CNN
from torchmetrics import MeanSquaredError, F1Score, Precision, Recall, AUROC

class ForestModel(pl.LightningModule):

    def __init__(self, input_width, lr, loss_fn, architecture, dropout=0, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()

        if architecture == "VGG":
            self.model = compile_VGG_CNN(input_width=input_width)
        else:
            self.model = compile_original_2D_CNN(input_width=input_width, dropout=dropout)
        
        if loss_fn == "BCEWithLogitsLoss":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.MSELoss()

        '''
        self.mse_metric = MeanSquaredError()
        self.rmse_metric = MeanSquaredError(squared=False)
        self.f1_metric = F1Score(task="binary")
        self.precision_metric = Precision(task="binary")
        self.recall_metric = Recall(task="binary")
        self.auroc_metric = AUROC(task="binary")
        '''

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.lr = lr
        self.weight_decay = weight_decay

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
        if self.loss_fn._get_name() != 'BCEWithLogitsLoss':
            output = torch.sigmoid(output)
        
        loss = self.loss_fn(output, target)

        metrics_batch = {}
        metrics_batch["loss"] = loss

        output = torch.sigmoid(output)

        '''
        metrics_batch["mse"] = self.mse_metric(output, target)
        metrics_batch["rmse"] = self.rmse_metric(output, target)
        metrics_batch["f1"] = self.f1_metric(output, target)
        metrics_batch["precision"] = self.precision_metric(output, target)
        metrics_batch["recall"] = self.recall_metric(output, target)
        metrics_batch["auroc"] = self.auroc_metric(output, target)
        '''

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
    
    def predict_step(self, batch, batch_idx):
        output = self.forward(batch[0])
        output = torch.sigmoid(output)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)