import argparse
import logging
import torch


from memory_maze import GridMazeWorld

def parse_args():
    parser = argparse.ArgumentParser("Train a maze agent using simple policy gradient method.")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument("--max-age", type=int, default=100)
    return parser.parse_args()


class SimpleLSTMAgent(torch.nn.Module):
    def __init__(self, observation_count, max_observation_value, action_count, hidden_size=128):
        """ A LSTM agent which can perform a single step or process a sequence of observations.
        It predicts an action and observation.
        :param observation_count:
        :param max_observation_value:
        :param action_count:
        :param hidden_size:
        """
        super(SimpleLSTMAgent, self).__init__()
        self.lstm = torch.nn.LSTM(observation_count, hidden_size)
        self.position_embedding = torch.nn.Embedding(max_observation_value, hidden_size)
        self.observation_embedding = torch.nn.Embedding(max_observation_value, hidden_size)

        self.action_prediction = torch.nn.Linear(128, action_count)




def main():
    args = parse_args()

    obsticle_fraction = 0.33
    grid_size = 12
    obsticle_count = (grid_size-2) * (grid_size-2) * obsticle_fraction
    food_source_count = 6

    environments = [GridMazeWorld(
        max_age=args.max_age,
        grid_size=grid_size,
        obstacle_count=obsticle_count,
        food_source_count=food_source_count,
        food_energy=12,
        initial_energy=30,
    ) for _ in range(args.batch_size)]


