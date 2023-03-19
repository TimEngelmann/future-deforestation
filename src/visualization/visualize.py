import torch

def visualize(path):

    predictions = torch.load(path + "test_predictions.pt")
    data = torch.load("data/processed/test_data.pt")

if __name__ == "__main__":
    path = "lightning_logs/version_7/"
    visualize(path)