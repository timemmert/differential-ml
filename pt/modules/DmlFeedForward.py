from typing import Callable, List, Tuple

import torch
from torch import nn

from pt.modules.DmlModule import DmlModule
from pt.modules.DmlLinear import DmlLinear

from differential_ml.pt.modules.device import global_device


class DmlFeedForward(DmlModule):

    def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            number_of_hidden_layers: int,
            hidden_layer_dimension: int,
            activation: Callable[[torch.Tensor], torch.Tensor]
            ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_layer_dimension = hidden_layer_dimension
        self.activation = activation

        self.input_layer = DmlLinear(self.input_dimension, self.hidden_layer_dimension, nn.Identity())
        self.hidden_layers = nn.ModuleDict(
            {
                self.get_layer_name(hidden_layer_number): DmlLinear(
                    self.hidden_layer_dimension,
                    self.hidden_layer_dimension,
                    self.activation,
                ) for hidden_layer_number in range(self.number_of_hidden_layers)
            }
        )
        self.output_layer = DmlLinear(self.hidden_layer_dimension, self.output_dimension, self.activation)

    @property
    def layers_as_list(self) -> List[DmlLinear]:
        # noinspection PyTypeChecker
        return [self.input_layer] + [hidden_layer for hidden_layer in self.hidden_layers.values()] + [self.output_layer]

    @staticmethod
    def get_layer_name(hidden_layer_number: int) -> str:
        return f'HiddenLayer{hidden_layer_number}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_outputs(x)[-1]

    def forward_with_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = [x]
        for layer in self.layers_as_list:
            outputs.append(layer(outputs[-1]))

        return outputs

    def greek(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        greek = torch.eye(outputs[-1].shape[1], device=global_device).repeat(outputs[-1].shape[0], 1, 1)
        for layer, output in reversed(list(zip(self.layers_as_list, outputs[:-1]))):
            greek = layer.greek(x=output, prev_greek=greek)

        return greek

    def forward_with_greek(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward_with_outputs(x)
        return outputs[-1], self.greek(outputs)
