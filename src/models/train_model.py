import torch

def get_data_loaders():
    # load datasets
    train_dataset = torch.load('data/processed/train_dataset.pt')
    val_dataset = torch.load('data/processed/val_dataset.pt')
    test_dataset = torch.load('data/processed/test_dataset.pt')

    # create the train, val and test dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model():
    train_loader, val_loader, test_loader = get_data_loaders()

if __name__ == "__main__":
    train_model()

