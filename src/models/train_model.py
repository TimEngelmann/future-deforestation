import torch
from torch.utils.data import Dataset

from transformers import SegformerImageProcessor
from transformers.image_utils import ChannelDimension

torch.manual_seed(17)

'''
Code based on: 
- https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''

class SegFormerDeforestationDataset(Dataset):
    """Forcasting Deforestation using SegFormer Dataset"""

    def __init__(self, dataset, feature_extractor):
        """
        Args:
            dataset: "train", "val", "test
        """
        self.features = torch.load(f'data/processed/{dataset}_features.pt')
        self.labels = torch.load(f'data/processed/{dataset}_labels.pt')
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(self.features[idx][:,:,:3], self.labels[idx], return_tensors="pt", data_format=ChannelDimension.LAST)

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs

def get_data_loaders():
    feature_extractor = SegformerImageProcessor(do_reduce_labels=True)

    # load datasets
    train_dataset = SegFormerDeforestationDataset("train", feature_extractor)
    val_dataset = SegFormerDeforestationDataset("val", feature_extractor)
    test_dataset = SegFormerDeforestationDataset("test", feature_extractor)

    # create the train, val and test dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model():
    train_loader, val_loader, test_loader = get_data_loaders()

if __name__ == "__main__":
    train_model()

