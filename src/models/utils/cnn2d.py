import torch
import numpy as np

def compile_original_2D_CNN(input_width=35, dropout=0):
    input_dim=4
    hidden_dim=[64, 128, 128]
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)]
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)]
    padding=[0, 0, 0, 0]
    maxpool_dim = 10
    dropout=dropout
    output_dim=1
    input_width=input_width

    layers = []

    # convolutional blocks
    width_conv_out = input_width
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
        width_conv_out = np.floor((width_conv_out + 2 * padding[i] - kernel_size[i][0])/stride[i][0] + 1)
    
    # intermediate layers
    stride_temp = np.floor(float(width_conv_out / maxpool_dim)).astype(np.int32)
    ksize_temp = int(stride_temp + (width_conv_out % maxpool_dim))
    layers.append(torch.nn.MaxPool2d(kernel_size=(ksize_temp, ksize_temp), stride=(stride_temp, stride_temp)))
    layers.append(torch.nn.Flatten())

    # linear layers
    ln_in = hidden_dim[-1] * maxpool_dim**2
    layers.append(torch.nn.Linear(ln_in, 100))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.BatchNorm1d(100))
    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(100, output_dim))

    model = torch.nn.Sequential(*layers)

    return model

def compile_VGG_CNN(input_width=400):
    input_dim=4
    hidden_dim=[64, 128, 256, 512, 512]
    kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    padding=[1, 1, 1, 1, 1, 1]
    maxpool_cnn = 2
    maxpool_dim = 10
    dropout=0.0
    output_dim=1
    input_width=35

    layers = []

    # convolutional blocks
    width_conv_out = input_width
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
        layers.append(torch.nn.MaxPool2d(maxpool_cnn, maxpool_cnn, 0))
        width_conv_out = np.floor((((width_conv_out + 2 * padding[i] - kernel_size[i][0])/stride[i][0] + 1))/maxpool_cnn)
    
    # intermediate layers
    layers.append(torch.nn.Flatten())

    # linear layers
    ln_in = int(hidden_dim[-1] * width_conv_out**2)
    layers.append(torch.nn.Linear(ln_in, 128))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(128, output_dim))

    model = torch.nn.Sequential(*layers)

    return model


