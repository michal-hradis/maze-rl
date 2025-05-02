import argparse
import cv2

import torch
import torch.nn.functional as F
from tqdm import tqdm

from memory_maze import GridMazeWorld
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser("Train a maze agent using simple policy gradient method.")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument("--max-age", type=int, default=50)
    parser.add_argument("--net-path", type=str, default="policy_epoch_100.pt")
    return parser.parse_args()


class AttnAggregator(nn.Module):
    """
    Attention‑style pooling that turns a set of token embeddings
    E ∈ ℝ^{B×T×K×D}  (B=batch, T=seq, K=observation_size, D=emb dim)
    into a single vector per observation:  ℝ^{B×T×D}.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)  # learnable query

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, K, D]  →  scores: [batch, seq, K, 1]
        scores = self.score(torch.tanh(x))
        α = torch.softmax(scores, dim=2)                  # weights along K
        pooled = (α * x).sum(dim=2)                       # [batch, seq, D]
        return pooled


class ConcatMLPAggregator(nn.Module):
    """
    ConcatMLP-style pooling that turns a set of token embeddings
    E ∈ ℝ^{B×T×K×D}  (B=batch, T=seq, K=observation_size, D=emb dim)
    into a single vector per observation:  ℝ^{B×T×D}.

    D - embed_dim
    K - embed_count
    """
    def __init__(self, embed_dim: int, embed_count: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_count = embed_count
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * embed_count, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, K, D]  →  x: [batch, seq, K*D]
        x = x.view(x.shape[0], x.shape[1], -1)
        # x: [batch, seq, K*D]  →  x: [batch, seq, D]
        x = self.mlp(x)
        return x


class MemoryPolicyNet(nn.Module):
    """
    Recurrent policy with per‑observation attention pooling and positional encodings.

    Parameters
    ----------
    vocab_size : int
        Size of token vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    observation_size : int
        Number of integer tokens in one observation (K).
    hidden_size : int
        LSTM hidden dimension.
    action_count : int
        Number of discrete actions.
    num_layers : int, default 1
        Stacked LSTM layers.
    pad_idx : int or None, default None
        Padding index for observations if used.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        observation_size: int,
        hidden_size: int,
        action_count: int,
        num_layers: int = 1,
        pad_idx: int = None,
    ):
        super().__init__()
        self.observation_size = observation_size

        # Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # Learnable positional encodings for the K tokens inside an observation
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)

        # Attention aggregator (K tokens → 1 vector)
        self.aggregator = ConcatMLPAggregator(embed_dim, observation_size)

        # Temporal memory across the seq dimension
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Policy head (logits)
        self.head = nn.Linear(hidden_size, action_count)
        self.hidden_state: tuple[torch.Tensor, torch.Tensor] = None

    def reset_state(self):
        self.hidden_state = None

    def forward(
        self,
        obs: torch.Tensor,                           # [batch, seq, K]  (Long)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : LongTensor[batch, seq, K]

        Returns
        -------
        logits : Tensor
        next_hidden : HiddenState
        """
        B, T, K = obs.shape
        assert K == self.observation_size, "Mismatch in observation_size"

        x = self.embedding(obs)                       # [B, T, K, D]
        x = x + self.pos_embed                        # broadcast (K, D) → (B,T,K,D)
        x = self.aggregator(x)                        # [B, T, D]

        # LSTM over the temporal dimension
        if self.hidden_state is None:
            # Initialize hidden state
            self.hidden_state = self.initial_state(B, device=x.device)

        out, self.hidden_state = self.lstm(x, self.hidden_state)       # out: [B, T, H]
        return self.head(out)

    def initial_state(self, batch_size: int, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0


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
    ) for i in range(args.batch_size)]

    observation_size = 10
    vocab_size      = 20
    embed_dim       = 768
    hidden_size     = 768
    action_count    = 8

    net = MemoryPolicyNet(
        vocab_size, embed_dim, observation_size,
        hidden_size, action_count
    )

    if args.net_path:
        checkpoint = torch.load(args.net_path)
        net.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    γ = 0.99  # discount
    entropy_coef = 0.01
    max_grad_norm = 1.0

    save_interval = 100

    for epoch in range(10000):
        with torch.no_grad():
            episode_actions, episode_observations, episode_rewards = get_episodes(args, device, environments, net)

        episode_observations = torch.cat(episode_observations, dim=1).to(device)  # [batch, seq, K]
        episode_actions = torch.stack(episode_actions, dim=1).to(device)  # [batch, seq]
        episode_rewards = torch.tensor(episode_rewards, dtype=torch.float).to(device).permute(1, 0)  # [batch, seq]

        net.reset_state()
        logits = net(episode_observations)

        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, A]
        act_log_probs = log_probs.gather(-1, episode_actions.unsqueeze(-1)).squeeze(-1)  # [B, T]

        with torch.no_grad():
            B, T = episode_rewards.shape
            returns = torch.zeros_like(episode_rewards)
            running = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                running = episode_rewards[:, t] + γ * running
                returns[:, t] = running

            # Simple baseline: mean return per trajectory
            baseline = returns.mean(dim=1, keepdim=True)  # [B, 1]
            advantages = returns - baseline  # [B, T]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(act_log_probs * advantages.detach()).mean()

        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        entropy_loss = -entropy_coef * entropy

        loss = policy_loss + entropy_loss

        # 7. Optimise
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

        average_rewards = episode_rewards.sum(1).mean().item()
        alive_lengths = (episode_rewards > 0).sum(1).float().mean()

        print(f"Epoch: {epoch}, Average reward: {average_rewards:.2f}, Live: {alive_lengths:.2f}, "
              f"Policy loss: {policy_loss.item():.4f}, Entropy loss: {entropy_loss.item():.4f}")

        if epoch % save_interval == 0:
            torch.save(net.state_dict(), f"policy_epoch_{epoch}.pt")
            print(f"Model saved at epoch {epoch}")


def get_episodes(args, device, environments, net, show=False):
    observations = [env.reset()[0] for env in environments]
    observations = torch.tensor(observations, dtype=torch.long).to(device)
    # add dimension from [batch, K] to [batch, 1, K]
    observations = observations.unsqueeze(1)
    net.reset_state()
    episode_rewards = []
    episode_observations = []
    episode_actions = []
    for i in range(args.max_age):
        episode_observations.append(observations)
        logits = net(observations).squeeze(1)  # [batch, 1, action_count]
        if show:  # select argmax action
            sampled_actions = logits.argmax(dim=-1)
        else:
            sampled_actions = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            sampled_actions = sampled_actions.squeeze(1)
        episode_actions.append(sampled_actions)

        # Take action
        # observations are: obsservation_token_list, reward, done, info
        results = [env.step(action) for env, action in zip(environments, sampled_actions.cpu().numpy())]

        if show:
            for env in environments[:1]:
                cv2.imshow(env.name, env.render(mode="image"))
                cv2.waitKey(50)

        observations, rewards, dones, infos = zip(*results)

        episode_rewards.append(rewards)

        observations = torch.tensor(observations, dtype=torch.long)
        observations = observations.unsqueeze(1).to(device)

        # Check if episode is done
        if all(dones):
            break

    if show:
        cv2.waitKey(500)
    return episode_actions, episode_observations, episode_rewards


if __name__ == "__main__":
    main()
