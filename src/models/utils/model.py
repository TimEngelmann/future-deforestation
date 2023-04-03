import torch
import pytorch_lightning as pl
from .cnn2d import compile_original_2D_CNN

class ForestModel(pl.LightningModule):

    def __init__(self, input_width):
        super().__init__()
        self.save_hyperparameters()

        self.model = compile_original_2D_CNN(input_width=input_width)
        
        self.loss_fn = torch.nn.MSELoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

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

        loss = self.loss_fn(output, target)

        metrics_batch = {}
        metrics_batch["loss"] = loss
        
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
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0)