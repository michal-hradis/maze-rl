

import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import json
from stable_baselines3.common.env_util import make_vec_env


import argparse
from functools import partial

from nets import CustomCNN

from environment import CustomEnv


def parseargs():
    parser = argparse.ArgumentParser(usage='Trains contrastive self-supervised training on artificial data.')
    parser.add_argument('--checkpoint', default=None, help='Load this checpoint at the start.')
    parser.add_argument('--log-name', default='default', help='Save results to this tensorboard log.')
    parser.add_argument('--run-game', action='store_true', help='Run and render the game without training.')

    parser.add_argument('--n-envs', default=32, type=int, help='Number of paralle environments.')
    parser.add_argument('--learning-params',
                        default='{"target_kl": 0.05, "learning_rate": 0.0004, "gamma": 0.95, "batch_size": 128}',
                        help='Json dictionary of params.')
    parser.add_argument('--env-params',
                        default='{"size": 12, "max_age": 100, "food_count": 3, "food_energy": 12, "obstacle_count": 36}',
                        help='Json dictionary of params.')
    parser.add_argument('--name', default="ppo_xxxx", help='Output checkpoint and tensorflow name.')

    args = parser.parse_args()
    return args


def build_env(env_constructor):
    env = env_constructor()
    check_env(env)
    return env


def build_rl(args, env):
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    n_steps = 2048 * 4 // args.n_envs
    learning_params = json.loads(args.learning_params)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log='PPO_LOG', policy_kwargs=policy_kwargs, n_steps=n_steps,
                **learning_params)
    if args.checkpoint:
        model.set_parameters(args.checkpoint)
    return model


def try_model(env, model):
    for i in range(1, 10000):
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, _ = env.step(model.predict(obs)[0])
            img = env.render(mode='img')
            cv2.imshow('game', img)
            key = cv2.waitKey(20)
            if key == 27:
                return


def main():
    args = parseargs()
    env_params = json.loads(args.env_params)
    env_constructor = partial(CustomEnv, **env_params)

    if args.run_game:
        env = build_env(env_constructor)
        agent = build_rl(args, env)
        try_model(env, agent)
        exit(-1)

    env = make_vec_env(env_constructor, n_envs=args.n_envs)
    agent = build_rl(args, env)

    for i in range(5000):
        steps_to_learn = 100000
        agent.learn(total_timesteps=steps_to_learn, reset_num_timesteps=False, tb_log_name=args.log_name)
        agent.save(f"{args.log_name}-{i*steps_to_learn}")


if __name__ == '__main__':
    main()


'''
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
        self.rec_count = 6
        self.input_block = nn.Sequential(nn.Conv2d(n_input_channels, self.inner_channels, kernel_size=3, padding=1, stride=1), nn.LeakyReLU())
        self.rec_block = nn.Sequential(nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]),
                                       nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1, stride=1),
                                       nn.LeakyReLU())
        self.output = nn.Sequential(
            nn.LeakyReLU(), nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1, stride=1),
            nn.LayerNorm([features_dim, observation_space.shape[1], observation_space.shape[2]]), nn.LeakyReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.input_block(observations)
        #print_info(x)
        for i in range(self.rec_count):
            x = x + self.rec_block(x)
            #print_info(x)
        x = self.output(x)
        x = x * observations[:, 2:, :, :]
        x = th.sum(x, dim=(2, 3))
        #print_info(x)
        return x'''