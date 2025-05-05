#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

from opacus.validators import ModuleValidator
from torch import nn

from src.picai_baseline.unet.training_setup.neural_networks.unets import UNet
import torch


def neural_network_for_run(args, device: torch.device | None = None):
    """Select neural network architecture for given run"""

    if args.model_type == 'unet':
        model = UNet(
            spatial_dims=len(args.image_shape),
            in_channels=args.num_channels,
            out_channels=args.num_classes,
            strides=args.model_strides,
            channels=args.model_features
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    model = prepare_dp_unet(model)
    if device:
        model = model.to(device)
    print("Loaded Neural Network Arch.:", args.model_type)
    return model


def prepare_dp_unet(model):
    """Prepare UNet model for differential privacy training."""
    print("Original model architecture:")
    #print(model)

    # First, replace unsupported layers
    model = replace_unsupported_layers(model)

    print("Modified model architecture:")
    #print(model)

    # Validate the model; returns a list of error messages (empty if no issues)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("Model has compatibility issues, fixing with ModuleValidator...")
        model = ModuleValidator.fix(model)

        # Re-validate after fixing
        post_errors = ModuleValidator.validate(model, strict=False)
        if post_errors:
            print("WARNING: Model still has issues with DP compatibility:")
            for err in post_errors:
                print(f"  • {err}")
        else:
            print("Model is now DP-compatible.")
    else:
        print("Model is already DP-compatible.")

    return model


def replace_unsupported_layers(model):
    """
    Replace layers unsupported by Opacus with compatible alternatives.

    Args:
        model: The original UNet model

    Returns:
        Modified model compatible with Opacus
    """
    # Recursively replace PReLU with LeakyReLU and ConvTranspose3d with alternatives
    for name, module in list(model.named_children()):
        # Replace PReLU with LeakyReLU (which is supported by Opacus)
        if isinstance(module, nn.PReLU):
            # LeakyReLU with similar negative_slope to PReLU
            # Default PReLU init is 0.25, so we use this as negative_slope
            model._modules[name] = nn.LeakyReLU(negative_slope=0.25)
            #print(f"Replaced {name} PReLU with LeakyReLU")

        # For ConvTranspose3d layers, we need to check if we can replace them
        # with a compatible alternative using Upsample + Conv3d
        elif isinstance(module, nn.ConvTranspose3d):
            # Get the parameters of the ConvTranspose3d
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding

            # Create a replacement using Upsample + Conv3d
            replacement = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='nearest'),
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,  # No stride after upsampling
                    padding=padding
                )
            )

            # Replace the module
            model._modules[name] = replacement
            #print(f"Replaced {name} ConvTranspose3d with Upsample + Conv3d")

        # Recursively handle nested modules
        elif len(list(module.children())) > 0:
            replace_unsupported_layers(module)

    return model
