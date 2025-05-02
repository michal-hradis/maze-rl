import argparse

import torch

from memory_maze import GridMazeWorld
from simple_training import MemoryPolicyNet, get_episodes


def parse_args():
    parser = argparse.ArgumentParser("Read a network and run it on a maze.")
    parser.add_argument("--max-age", type=int, default=50)
    parser.add_argument("--net-path", type=str, default="policy_epoch_100.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    obstacle_fraction = 0.25
    grid_size = 11
    obstacle_count = (grid_size - 2) * (grid_size - 2) * obstacle_fraction
    food_source_count = 10
    food_energy = 10
    initial_energy = 20

    environments = [GridMazeWorld(
        max_age=args.max_age,
        grid_size=grid_size,
        obstacle_count=obstacle_count,
        food_source_count=food_source_count,
        food_energy=food_energy,
        initial_energy=initial_energy,
        name=str(i),
    ) for i in [0]]

    observation_size = 10
    vocab_size      = 20
    embed_dim       = 768
    hidden_size     = 768
    action_count    = 8

    net = MemoryPolicyNet(
        vocab_size, embed_dim, observation_size,
        hidden_size, action_count
    )

    # Load the network state
    checkpoint = torch.load(args.net_path)
    net.load_state_dict(checkpoint)
    net.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(10000):
        with torch.no_grad():
            episode_actions, episode_observations, episode_rewards = get_episodes(args, device, environments, net, show=True)



if __name__ == "__main__":
    main()
