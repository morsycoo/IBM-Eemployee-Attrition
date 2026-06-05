import torch
import torch.nn as nn


class TunableEmployeeAttritionNN(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden1,
        hidden2,
        dropout
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden1
            ),

            nn.BatchNorm1d(
                hidden1
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden1,
                hidden2
            ),

            nn.BatchNorm1d(
                hidden2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden2,
                1
            )
        )

    def forward(
        self,
        x
    ):
        return self.network(
            x
        )