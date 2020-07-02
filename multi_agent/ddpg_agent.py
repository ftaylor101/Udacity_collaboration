import numpy as np
import random
import copy
from collections import namedtuple, deque
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim

from multi_agent.models import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """
    A DDPG Agent that interacts and learns from the environment.

    Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, state_size: int = 24, action_size: int = 2, lr_actor: float = 1e-4, lr_critic: float = 1e-4,
                 replay_buffer_size: int = 1e6, gamma: float = 0.99, batch_size: int = 128, random_seed: int = 10,
                 soft_update_tau: float = 1e-3):
        """
        Initialise an Agent object.

        :param state_size: the size of the environment observation
        :param action_size: the size of action taken
        :param lr_actor: learning rate for actor
        :param lr_critic: learning rate for critic
        :param replay_buffer_size: replay buffer size
        :param gamma: reward discount
        :param batch_size: size of experience sample from replay buffer
        :param random_seed: random seed
        :param soft_update_tau: size of target network update
        """
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.tau = soft_update_tau
        self.batch_size = batch_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size=state_size, action_size=action_size).to(device=device)
        self.actor_target = Actor(state_size=state_size, action_size=action_size).to(device=device)
        self.actor_optimiser = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size=state_size, action_size=action_size).to(device=device)
        self.critic_target = Critic(state_size=state_size, action_size=action_size).to(device=device)
        self.critic_optimiser = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, replay_buffer_size, batch_size, random_seed)

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        Save experience in replay memory, and use random sample from buffer to learn.

        :param state: the currently observed environment state
        :param action: the action picked based on the current state
        :param reward: the reward received for performing the action
        :param next_state: the state transitioned to based on performing the action
        :param done: if the episode is complete or not
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences=experiences, gamma=self.gamma)

    def act(self, state: np.ndarray, noise_t: float = 0.0) -> np.ndarray:
        """
        Returns actions for given state as per current policy.

        :param state: the current state of the environment
        :param noise_t: a noise factor to adjust how much noise is added to the action
        """
        state = torch.from_numpy(state).float().to(device=device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample() * noise_t
        return np.clip(action, -1, 1)

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              gamma: float):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences: tuple of (s, a, r, s', done) tuples
        :param gamma: discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)).detach()
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters. Every learning step the target network is updated to bring its parameters nearer
        by a factor TAU to those of the improving local network.

        If TAU = 1 the target network becomes a copy of the local network.
        If TAU = 0 the target network is not updated.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: weights will be copied from
        :param target_model: weights will be copied to
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """
    Ornstein-Uhlenbeck process. Used to add noise to the action selection.
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialise parameters and noise process.

        :param size: size of the vector to add noise to
        :param seed: random seed
        :param mu: OU constant
        :param theta: OU parameter
        :param sigma: OU parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = copy.copy(self.mu)
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """
        Update internal state and return it as a noise sample.

        :return: the noise to be added
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        """
        Initialize a ReplayBuffer object.

        :param action_size: the size of the action space
        :param buffer_size: the maximum size of the buffer
        :param batch_size: the size of each training batch
        :param seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
