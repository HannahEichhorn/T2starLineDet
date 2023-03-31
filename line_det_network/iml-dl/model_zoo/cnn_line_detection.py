import torch
import torch.nn as nn


class RealValCNNLineDetNew(nn.Module):
    """ Convolutional network to classify kspace lines

    Input to the network is multi-echo kspace data, where real and imaginary
    parts are on two separate channels. Output is a mask classifying each line as
    artefact-free or corrupted.

    """
    def __init__(self, activation="relu", input_dim=2, input_size=[12, 92, 112],
                 crop_readout=False, output_size=[12, 92], first_filters=16, last_filters=False,
                 kernel_size=5, num_layer=4, dropout=False, use_bias=True,
                 dtype=torch.float64, normalization=None, **kwargs):

        super(RealValCNNLineDetNew, self).__init__()
        # get the correct conv operator
        conv_layer = nn.Conv3d

        if activation == "relu":
            act_layer = nn.ReLU

        if normalization == "BatchNorm":
            norm_layer = nn.BatchNorm3d

        filters = []
        for i in range(1, num_layer // 2 + 1):
            filters.append(first_filters * i)
            filters.append(first_filters * i)
        if num_layer % 2 == 1:
            filters.append(first_filters * (i + 1))

        padding = kernel_size // 2
        pool_kernel_size = []
        pool_stride = []
        nr = num_layer + 1 if last_filters is not False else num_layer
        for i in range(0, min(3, nr)):
            pool_kernel_size.append((2, 1, 2))
            pool_stride.append((2, 1, 2))
        if num_layer > 3:
            for i in range(3, nr):
                pool_kernel_size.append((1, 1, 2))
                pool_stride.append((1, 1, 2))

        # create layers
        self.ops = []
        self.ops.append(conv_layer(in_channels=input_dim,
                                   out_channels=filters[0],
                                   kernel_size=kernel_size, padding=padding,
                                   bias=use_bias, dtype=dtype, **kwargs))

        if normalization == "BatchNorm":
            self.ops.append(norm_layer(filters[0], dtype=dtype))
        self.ops.append(act_layer())
        if dropout is not False:
            self.ops.append(nn.Dropout3d(p=dropout))
        self.ops.append(nn.MaxPool3d(kernel_size=pool_kernel_size[0],
                                       stride=pool_stride[0]))

        for i in range(1, num_layer):
            self.ops.append(conv_layer(in_channels=filters[i-1],
                                       out_channels=filters[i],
                                       kernel_size=kernel_size, padding=padding,
                                       bias=use_bias, dtype=dtype, **kwargs))

            if normalization == "BatchNorm":
                self.ops.append(norm_layer(filters[i], dtype=dtype))
            self.ops.append(act_layer())
            if dropout is not False:
                self.ops.append(nn.Dropout3d(p=dropout))
            self.ops.append(nn.MaxPool3d(kernel_size=pool_kernel_size[i],
                                           stride=pool_stride[i]))

        if last_filters is not False:
            self.ops.append(conv_layer(in_channels=filters[num_layer-1],
                                       out_channels=last_filters[0],
                                       kernel_size=kernel_size, padding=padding,
                                       bias=use_bias, dtype=dtype, **kwargs))

            if normalization == "BatchNorm":
                self.ops.append(norm_layer(last_filters[0], dtype=dtype))
            self.ops.append(act_layer())
            if dropout is not False:
                self.ops.append(nn.Dropout3d(p=dropout))
            self.ops.append(nn.MaxPool3d(kernel_size=pool_kernel_size[-1],
                                         stride=pool_stride[-1]))

            if len(last_filters) > 1:
                for i in range(1, len(last_filters)):
                    self.ops.append(conv_layer(in_channels=last_filters[i-1],
                                               out_channels=last_filters[i],
                                               kernel_size=kernel_size, padding=padding,
                                               bias=use_bias, dtype=dtype, **kwargs))

                    if normalization == "BatchNorm":
                        self.ops.append(norm_layer(last_filters[i], dtype=dtype))
                    self.ops.append(act_layer())
                    if dropout is not False:
                        self.ops.append(nn.Dropout3d(p=dropout))

        self.ops = torch.nn.Sequential(*self.ops)

        # determine size of feature maps after convolutions for fc layer:
        size1 = input_size[0]
        size2 = crop_readout if crop_readout is not False else input_size[2]
        for j in range(0, min(3, nr)):
            size1 = size1 // 2
        for j in range(0, nr):
            size2 = size2 // 2
        num_filters = filters[num_layer-1] if last_filters is False else last_filters[-1]

        self.fc = torch.nn.Linear(num_filters*input_size[1]*size1*size2,
                                  output_size[0]*output_size[1], dtype=dtype)
        if dropout is not False:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        self.Sigmoid = torch.nn.Sigmoid()

        self.apply(self.weight_init)


    def weight_init(self, module):
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)


    def forward(self, x):
        x = self.ops(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.Sigmoid(x)

        return x

