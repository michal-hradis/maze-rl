import torch.nn as nn
import torch
import gym


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def print_info(t):
    print(t.shape, torch.std(t).item(), t.min().item(), t.max().item())


class SeparableConv(torch.nn.Module):
    def __init__(self, features_dim: int = 64):
        super(SeparableConv, self).__init__()

        self.conv1 = nn.Conv2d(features_dim, features_dim, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.conv2 = nn.Conv2d(features_dim, features_dim, kernel_size=(3, 1), padding=(1, 0), stride=1)

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        # OUTPUT = ((INPUT - KERNEL + 2*PADDING) / STRIDE) + 1
        # OUTPUT = ((15-3+2*1)/1)+1
        # OUTPUT = 15

        self.inner_channels = features_dim
        self.rec_count = 12
        self.input_block = nn.Sequential(nn.Conv2d(n_input_channels, self.inner_channels, kernel_size=3, padding=1, stride=1), nn.LeakyReLU())
        '''self.rec_block = nn.Sequential(nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]),
                                       nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1, stride=1),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1, stride=1)
                                       )'''
        self.rec_block = nn.Sequential(
            nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]),
            SeparableConv(self.inner_channels),
            nn.LeakyReLU(),
            SeparableConv(self.inner_channels),
            )
        self.output = nn.Sequential(nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]), nn.LeakyReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_block(observations)
        for i in range(self.rec_count):
            x = x + self.rec_block(x)
            x = x * (1-observations[:, :1])
        x = self.output(x)
        x = x * observations[:, 2:3, :, :] - x * observations[:, 3:4, :, :]
        x = torch.sum(x, dim=(2, 3)) * 0.25
        return x


class CustomCNN_v2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNN_v2, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        # OUTPUT = ((INPUT - KERNEL + 2*PADDING) / STRIDE) + 1
        # OUTPUT = ((15-3+2*1)/1)+1
        # OUTPUT = 15

        self.inner_channels = features_dim
        self.rec_count = 3
        self.input_block = nn.Sequential(nn.Conv2d(n_input_channels, self.inner_channels, kernel_size=3, padding=1, stride=1), nn.LeakyReLU())
        self.res_blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm([self.inner_channels, observation_space.shape[1], observation_space.shape[2]]),
                           nn.LeakyReLU(),
                           nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1, stride=1),
                          ),
             nn.Sequential(nn.LayerNorm([self.inner_channels, observation_space.shape[1], observation_space.shape[2]]),
                           nn.LeakyReLU(),
                           nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=2, stride=1, dilation=2),
                           ),
             nn.Sequential(nn.LayerNorm([self.inner_channels, observation_space.shape[1], observation_space.shape[2]]),
                           nn.LeakyReLU(),
                           nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=3, stride=1, dilation=3),
                           )
             ]
        )
        self.norm_blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]),
                                    nn.LeakyReLU()),
             nn.Sequential(nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]),
                           nn.LeakyReLU()),
             nn.Sequential(nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]),
                           nn.LeakyReLU()),
             ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_block(observations)
        for res_block, norm_block in zip(self.res_blocks, self.norm_blocks):
            for i in range(self.rec_count):
                x = x + res_block(x)
            x = norm_block(x)
        x = self.output(x)
        x = x * observations[:, 2:, :, :]
        x = torch.sum(x, dim=(2, 3))
        return x