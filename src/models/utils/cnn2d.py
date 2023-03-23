import torch

def compile_2D_CNN():
    input_dim=4
    hidden_dim=[128, 64, 64, 32]
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)]
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)]
    padding=[0, 0, 0, 0]
    dropout=0.4
    output_dim=1

    layers = []

    # convolutional blocks
    for i in range(len(hidden_dim)):
        layer_input_dim = input_dim if i == 0 else hidden_dim[i-1]
        layers.append(
            torch.nn.Conv2d(
            layer_input_dim,
            hidden_dim[i],
            kernel_size[i],
            stride=stride[i],
            padding=padding[i],
        ))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm2d(hidden_dim[i]))

    layers.append(torch.nn.MaxPool2d(2, 2, 0))
    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Flatten())

    lin_in = 512 # maybe find way to calculate this
    layers.append(torch.nn.Linear(lin_in, 100))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.BatchNorm1d(100))
    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(100, output_dim))
    layers.append(torch.nn.Sigmoid())

    model = torch.nn.Sequential(*layers)

    return model