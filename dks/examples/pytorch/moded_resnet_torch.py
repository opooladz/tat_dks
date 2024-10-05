'''
first pass on pytorch version of this code
'''
import math
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dks.pytorch import activation_transform
from dks.pytorch import parameter_sampling_functions

FloatStrOrBool = Union[str, float, bool]

class BlockV1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Sequence[int]],
        use_projection: bool,
        bottleneck: bool,
        use_batch_norm: bool,
        activation: Callable[[torch.Tensor], torch.Tensor],
        shortcut_weight: Optional[float],
    ):
        super().__init__()
        
        self.use_projection = use_projection
        self.use_batch_norm = use_batch_norm
        self.shortcut_weight = shortcut_weight
        self.activation = activation

        if self.use_projection and self.shortcut_weight != 0.0:
            self.proj_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_batch_norm)
            if use_batch_norm:
                self.proj_bn = nn.BatchNorm2d(out_channels)

        channel_div = 4 if bottleneck else 1
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # First convolution
        self.conv_layers.append(nn.Conv2d(
            in_channels, out_channels // channel_div,
            kernel_size=1 if bottleneck else 3,
            stride=1 if bottleneck else stride,
            padding=0 if bottleneck else 1,
            bias=not use_batch_norm
        ))
        if use_batch_norm:
            self.bn_layers.append(nn.BatchNorm2d(out_channels // channel_div))

        # Second convolution
        self.conv_layers.append(nn.Conv2d(
            out_channels // channel_div, out_channels // channel_div,
            kernel_size=3, stride=stride if bottleneck else 1,
            padding=1, bias=not use_batch_norm
        ))
        if use_batch_norm:
            self.bn_layers.append(nn.BatchNorm2d(out_channels // channel_div))

        # Third convolution (for bottleneck)
        if bottleneck:
            self.conv_layers.append(nn.Conv2d(
                out_channels // channel_div, out_channels,
                kernel_size=1, stride=1, bias=not use_batch_norm
            ))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(out_channels))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                parameter_sampling_functions.scaled_uniform_orthogonal_(m.weight, delta=True)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shortcut = x

        if self.use_projection and self.shortcut_weight != 0.0:
            shortcut = self.proj_conv(shortcut)
            if self.use_batch_norm:
                shortcut = self.proj_bn(shortcut)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            if i < len(self.conv_layers) - 1:  # Don't apply activation on last layer
                x = self.activation(x)

        if self.shortcut_weight is None:
            return self.activation(x + shortcut)
        elif self.shortcut_weight != 0.0:
            return self.activation(
                math.sqrt(1 - self.shortcut_weight ** 2) * x +
                self.shortcut_weight * shortcut
            )
        else:
            return x

class ModifiedResNet(nn.Module):
    CONFIGS = {
        18: {"blocks_per_group": (2, 2, 2, 2), "bottleneck": False, "channels_per_group": (64, 128, 256, 512), "use_projection": (False, True, True, True)},
        34: {"blocks_per_group": (3, 4, 6, 3), "bottleneck": False, "channels_per_group": (64, 128, 256, 512), "use_projection": (False, True, True, True)},
        50: {"blocks_per_group": (3, 4, 6, 3), "bottleneck": True, "channels_per_group": (256, 512, 1024, 2048), "use_projection": (True, True, True, True)},
        101: {"blocks_per_group": (3, 4, 23, 3), "bottleneck": True, "channels_per_group": (256, 512, 1024, 2048), "use_projection": (True, True, True, True)},
        152: {"blocks_per_group": (3, 8, 36, 3), "bottleneck": True, "channels_per_group": (256, 512, 1024, 2048), "use_projection": (True, True, True, True)},
    }

    def __init__(
        self,
        num_classes: int,
        depth: int,
        use_batch_norm: bool = False,
        shortcut_weight: Optional[float] = 0.0,
        activation_name: str = "softplus",
        dropout_rate: float = 0.0,
        transformation_method: str = "DKS",
        dks_params: Optional[Mapping[str, FloatStrOrBool]] = None,
        tat_params: Optional[Mapping[str, FloatStrOrBool]] = None,
    ):
        super().__init__()

        self.depth = depth
        self.use_batch_norm = use_batch_norm
        self.shortcut_weight = shortcut_weight

        config = self.CONFIGS[depth]
        self.bottleneck = config["bottleneck"]
        self.blocks_per_group = config["blocks_per_group"]
        self.channels_per_group = config["channels_per_group"]
        self.use_projection = config["use_projection"]

        # Prepare activation function
        def subnet_max_func(x, r_fn):
            return self.subnet_max_func(x, r_fn)

        act_dict = activation_transform.get_transformed_activations(
            [activation_name], method=transformation_method,
            dks_params=dks_params, tat_params=tat_params,
            subnet_max_func=subnet_max_func
        )
        self.activation = act_dict[activation_name]

        # Initial convolution
        self.initial_conv = nn.Conv2d(3, self.channels_per_group[0], kernel_size=7, stride=2, padding=3, bias=not use_batch_norm)
        if use_batch_norm:
            self.initial_bn = nn.BatchNorm2d(self.channels_per_group[0])

        # ResNet blocks
        self.layers = nn.ModuleList()
        in_channels = self.channels_per_group[0]
        for i, (block_channels, num_blocks) in enumerate(zip(self.channels_per_group, self.blocks_per_group)):
            layer = self._make_layer(in_channels, block_channels, num_blocks, stride=1 if i == 0 else 2, use_projection=self.use_projection[i])
            self.layers.append(layer)
            in_channels = block_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels_per_group[-1], num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                parameter_sampling_functions.scaled_uniform_orthogonal_(m.weight, delta=True)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                parameter_sampling_functions.scaled_uniform_orthogonal_(m.weight, delta=False)
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_projection):
        layers = []
        layers.append(BlockV1(in_channels, out_channels, stride, use_projection, self.bottleneck,
                              self.use_batch_norm, self.activation, self.shortcut_weight))
        for _ in range(1, num_blocks):
            layers.append(BlockV1(out_channels, out_channels, 1, False, self.bottleneck,
                                  self.use_batch_norm, self.activation, self.shortcut_weight))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        if self.use_batch_norm:
            x = self.initial_bn(x)
        x = self.activation(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def subnet_max_func(self, x, r_fn, resnet_v2=True):
        return subnet_max_func_impl(x, r_fn, self.depth, self.shortcut_weight, resnet_v2=resnet_v2)

def subnet_max_func_impl(x, r_fn, depth, shortcut_weight, resnet_v2=True):
    """The subnetwork maximizing function of the modified ResNet model."""

    CONFIGS = {
        18: {"blocks_per_group": [2, 2, 2, 2], "bottleneck": False, "use_projection": [False, True, True, True]},
        34: {"blocks_per_group": [3, 4, 6, 3], "bottleneck": False, "use_projection": [False, True, True, True]},
        50: {"blocks_per_group": [3, 4, 6, 3], "bottleneck": True, "use_projection": [True, True, True, True]},
        101: {"blocks_per_group": [3, 4, 23, 3], "bottleneck": True, "use_projection": [True, True, True, True]},
        152: {"blocks_per_group": [3, 8, 36, 3], "bottleneck": True, "use_projection": [True, True, True, True]},
    }

    blocks_per_group = CONFIGS[depth]["blocks_per_group"]
    bottleneck = CONFIGS[depth]["bottleneck"]
    use_projection = CONFIGS[depth]["use_projection"]

    if bottleneck and resnet_v2:
        res_fn = lambda z: r_fn(r_fn(r_fn(z)))
    elif (not bottleneck and resnet_v2) or (bottleneck and not resnet_v2):
        res_fn = lambda z: r_fn(r_fn(z))
    else:
        res_fn = r_fn

    res_branch_subnetwork = res_fn(x)

    for i in range(4):
        for j in range(blocks_per_group[i]):
            res_x = res_fn(x)

            if j == 0 and use_projection[i] and resnet_v2:
                shortcut_x = r_fn(x)
            else:
                shortcut_x = x

            x = (shortcut_weight ** 2) * shortcut_x + (1.0 - shortcut_weight ** 2) * res_x

            if not resnet_v2:
                x = r_fn(x)

    x = r_fn(x)

    # Use the built-in max function for scalar values
    return max(x, res_branch_subnetwork)

def create_resnet18(num_classes=1000, use_batch_norm=True, shortcut_weight=0.0,
                    activation_name="relu", dropout_rate=0.0,
                    transformation_method="DKS", dks_params=None, tat_params=None):
    
    if dks_params is None:
        dks_params = {}
    if tat_params is None:
        tat_params = {}
    
    print(f"Creating ResNet18 with transformation_method: {transformation_method}")
    print(f"TAT params: {tat_params}")

    model = ModifiedResNet(
        num_classes=num_classes,
        depth=18,  # This specifies ResNet18
        use_batch_norm=use_batch_norm,
        shortcut_weight=shortcut_weight,
        activation_name=activation_name,
        dropout_rate=dropout_rate,
        transformation_method=transformation_method,
        dks_params=dks_params,
        tat_params=tat_params
    )
    
    return model

# Example usage:
if __name__ == "__main__":
    # First, create a DKS version (this should work as before)
    resnet18_dks = create_resnet18(
        num_classes=1000,
        use_batch_norm=True,
        shortcut_weight=0.0,
        activation_name="leaky_relu",
        dropout_rate=0.2,
        transformation_method="DKS",
        # Don't specify any DKS parameters
    )

    # You can now use this model for training or inference
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
    output = resnet18_dks(input_tensor)
    print("DKS output shape:", output.shape)  # Should print torch.Size([1, 1000])
    
    tat_params = {
        "c_val_0_target": 0.9,  # This should be between 0.0 and 1.0
        "c_curve_target": 0.3   # This should be greater than 0.0
    }
    # Now, let's try creating a TAT version without specific parameters
    resnet18_tat = create_resnet18(
        num_classes=1000,
        use_batch_norm=False,  # TAT is typically used without batch normalization
        shortcut_weight=0.0,
        activation_name="leaky_relu",  # TAT with Leaky ReLU is recommended
        dropout_rate=0.2,
        transformation_method="TAT",
        tat_params=tat_params  # Now specifying TAT parameters
    )

    # You can now use this model for training or inference
    output_tat = resnet18_tat(input_tensor)
    print("TAT output shape:", output_tat.shape)  # Should print torch.Size([1, 1000])
