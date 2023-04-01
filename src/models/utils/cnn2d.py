import torch
import numpy as np

def spp_layer(input_, levels=[6]):
    shape = input_.shape[-3:]
    pyramid = []
    for n in levels:
        if (n > shape[1]) or (n > shape[2]):
            raise ("Levels are not correct!")
        else:
            stride_1 = np.floor(float(shape[1] / n)).astype(np.int32)
            stride_2 = np.floor(float(shape[2] / n)).astype(np.int32)
            ksize_1 = stride_1 + (shape[1] % n)
            ksize_2 = stride_2 + (shape[2] % n)
            pool_layer = torch.nn.MaxPool2d(
                kernel_size=(ksize_1, ksize_2), stride=(stride_1, stride_2)
            )
            pool = pool_layer(input_)
            pool = pool.view(-1, shape[0] * n * n)
        pyramid.append(pool)
    spp_pool = torch.cat(pyramid, 1)
    return spp_pool

def compile_original_2D_CNN(input_width=400):
    input_dim=4
    hidden_dim=[64, 128, 128]
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)]
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)]
    padding=[0, 0, 0, 0]
    maxpool_dim = 10
    dropout=0.0
    output_dim=1
    input_width=400

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
    layers.append(torch.nn.Sigmoid())

    model = torch.nn.Sequential(*layers)

    return model

def compile_2D_CNN():
    input_dim=4
    hidden_dim=[128, 64, 64, 32]
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)]
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)]
    padding=[0, 0, 0, 0]
    dropout=0.2
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

    lin_in = 288800 # maybe find way to calculate this
    layers.append(torch.nn.Linear(lin_in, 100))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.BatchNorm1d(100))
    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(100, output_dim))
    layers.append(torch.nn.Sigmoid())

    model = torch.nn.Sequential(*layers)

    return model

def compile_simple_2D_CNN():
    input_dim = 4
    output_dim = 1

    layers = []
    layers.append(torch.nn.Conv2d(input_dim, 16, 5, 2, 0))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Conv2d(16, 32, 3, 1, 0))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Conv2d(32, 64, 3, 1, 0))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.MaxPool2d(2, 2, 0))
    layers.append(torch.nn.Dropout(0.25))
    layers.append(torch.nn.Flatten())
    lin_in = 602176
    layers.append(torch.nn.Linear(lin_in, 128))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Dropout(0.5))
    layers.append(torch.nn.Linear(128, output_dim))
    layers.append(torch.nn.Sigmoid())

    model = torch.nn.Sequential(*layers)
    return model
