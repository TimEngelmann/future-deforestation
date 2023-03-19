import torch
import torchmetrics
import pytorch_lightning as pl
from .cnn2d import compile_2D_CNN

class ForestModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = compile_2D_CNN()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        self.auroc = torchmetrics.AUROC(task="binary")
        self.prec = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, features):
        logits = self.model(features)
        return logits
    
    def shared_step(self, batch, stage):

        features = batch[0]
        target = batch[1]
        
        logits = self.forward(features)
        logits = logits.squeeze()

        loss = self.loss_fn(logits, target)

        probs = logits.sigmoid()
        preds = (probs > 0.5).float()

        metrics_batch = {}
        metrics_batch["loss"] = loss
        metrics_batch[f"auc"] = self.auroc(probs, target)
        metrics_batch[f"precision"] = self.prec(preds, target)
        metrics_batch[f"recall"] = self.recall(preds, target)
        metrics_batch[f"f1"] = self.f1(preds, target)

        '''
        for i in range(preds.shape[1]):
            metrics_batch[f"auc_{i}"] = self.auroc(probs[:,i], target[:,i])
            metrics_batch[f"precision_{i}"] = self.prec(preds[:,i], target[:,i])
            metrics_batch[f"recall_{i}"] = self.recall(preds[:,i], target[:,i])
            metrics_batch[f"f1_{i}"] = self.f1(preds[:,i], target[:,i])
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
        logits = self.forward(batch[0])
        logits = logits.squeeze()
        probs = logits.sigmoid()
        return probs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.1)