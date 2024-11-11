#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class DuelingDQN(nn.Module):
    """Dueling Deep Q-learning network."""

    def __init__(self, in_channels, num_actions):
        """
        Parameters:
        -----------
        in_channels: number of channels in the input (e.g., stacked frames in an Atari game).
        num_actions: number of actions available in the environment.
        """
        super(DuelingDQN, self).__init__()
        
        # Shared convolutional feature extraction layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        
        # Flatten layer to connect convolutional output to fully connected layers
        self.flatten = nn.Flatten()
        
        # Value stream layers
        self.value_fc1 = nn.Linear(3136, 512)  # 3136 is calculated based on input dimensions
        self.value_fc2 = nn.Linear(512, 1)  # Outputs a single value for the state

        # Advantage stream layers
        self.advantage_fc1 = nn.Linear(3136, 512)
        self.advantage_fc2 = nn.Linear(512, num_actions)  # Outputs a separate advantage for each action

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Dueling DQN network.
        Parameters:
        -----------
        x: torch.Tensor
            Input tensor representing the state.
        Returns:
        --------
        torch.Tensor
            Q-values for each action.
        """
        # Pass through convolutional layers
        x = self.conv(x)
        x = self.flatten(x)

        # Value stream forward pass
        value = F.leaky_relu(self.value_fc1(x))
        value = self.value_fc2(value)

        # Advantage stream forward pass
        advantage = F.leaky_relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        # Combine Value and Advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
