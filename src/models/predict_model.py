import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel

def get_data_loaders(batch_size=64, num_workers=5, max_elements=None):
    val_dataset = DeforestationDataset("val", max_elements=max_elements)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = DeforestationDataset("test", max_elements=max_elements)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader, test_loader

def predict_model(input_path, output_path):

    val_loader, test_loader = get_data_loaders(max_elements=50000)
    
    model = ForestModel.load_from_checkpoint(input_path)
    trainer = pl.Trainer(
        accelerator='mps', 
        devices=1,
        logger=False
    )

    test_predictions =  trainer.predict(model, test_loader)
    test_predictions = torch.cat(test_predictions)
    torch.save(test_predictions, output_path + "test_predictions.pt")

    valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)

    print("----- Validation Metrics -----")
    print(valid_metrics)
    print("----- Testing Metrics -----")
    print(test_metrics)

if __name__ == "__main__":
    input_path = "lightning_logs/version_16/checkpoints/epoch=29-step=23460.ckpt"
    output_path = "lightning_logs/version_16/"
    predict_model(input_path, output_path)
