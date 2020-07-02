import numpy as np
from typing import List

from multi_agent.ddpg_agent import Agent


class MultiAgentDDPG:
    """
    A multi agent class that instantiates two separate DDPG agents for training in a collaborative environment.
    """

    def __init__(self, state_size: int = 24, action_size: int = 2, lr_actor: float = 1e-5, lr_critic: float = 1e-4,
                 replay_buffer_size: int = 1e6, gamma: float = 0.95, batch_size: int = 64, random_seed: int = 10,
                 soft_update_tau: float = 1e-3):
        """
        Constructor for the Multi Agent DDPG class that instantiates two DDPG agents with the necessary parameters.

        :param state_size: the environment state size for a single agent
        :param action_size: the environment action size for a single agent
        :param lr_actor: learning rate for the Actor model
        :param lr_critic: learning rate for the Critic model
        :param replay_buffer_size: the replay buffer size
        :param gamma: the reward discount factor
        :param batch_size: the size of the sample retrieved from the buffer
        :param random_seed: random seed
        :param soft_update_tau: the interpolation parameter to move model target weights towards model local weights
        """
        ddpg_agent = Agent(state_size=state_size, action_size=action_size, lr_actor=lr_actor, lr_critic=lr_critic,
                           replay_buffer_size=replay_buffer_size, gamma=gamma, batch_size=batch_size,
                           random_seed=random_seed, soft_update_tau=soft_update_tau)
        self.multi_agent_ddpg = [ddpg_agent, ddpg_agent]

    def act(self, states: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """
        Get the action to perform for each agent given their respective state.

        :param states: the states from both agents
        :param noise: noise to apply to each action
        :return: an action for each agent to apply to the environment
        """
        actions = []
        for i in range(2):
            a = self.multi_agent_ddpg[i].act(states[i], noise)
            actions.append(a)
        return np.array(actions)

    def update(self, states: np.ndarray, actions: np.ndarray, rewards: List[float], next_states: np.ndarray,
               dones: List[bool]):
        """
        Update the Actor Critic network of both agents by calling their respective step function to save their
        experience, sample from it, and learn.

        :param states: the currently observed environment state for both agents
        :param actions: the action picked based on the current state for both agents
        :param rewards: the reward received for performing the action for both agents
        :param next_states: the state transitioned to based on performing the action for both agents
        :param dones: if the episode is complete or not for both agents
        """
        for i in range(2):
            self.multi_agent_ddpg[i].step(states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i])
