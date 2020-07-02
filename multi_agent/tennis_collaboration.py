from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import torch
from multi_agent.multi_ddpg_agent import MultiAgentDDPG
import matplotlib.pyplot as plt
import os
import time


def train_maddpg(env: UnityEnvironment, agent: MultiAgentDDPG, number_of_episodes: int = 5000):
    brain_name = env.brain_names[0]

    score_deque = deque(maxlen=100)
    avg_rolling_score = []
    score_per_episode = []
    noise_scaler = 1.0  # scale action noise by this amount
    noise_decay = 0.995  # decay action noise by this amount

    for i_episode in range(1, number_of_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(2)
        done = False
        while not np.any(done):
            action = agent.act(state, noise_scaler)
            env_info = env.step(action)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done = env_info.local_done
            agent.update(state, action, rewards, next_states, done)

            noise_scaler *= noise_decay
            score += rewards
            state = next_states

        episode_score = np.max(score)
        score_deque.append(episode_score)
        avg_rolling_score.append(np.mean(score_deque))
        score_per_episode.append(episode_score)
        print(f"Episode {i_episode} Score: {episode_score}")
        print(f"Average score over 100 episodes: {np.mean(score_deque)}")

        if np.mean(score_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode - 100,
                                                                                         np.mean(score_deque)))
            for i in range(2):
                torch.save(agent.multi_agent_ddpg[i].actor_local.state_dict(), f"agent_{i}_actor_local_checkpoint.pth")
                torch.save(agent.multi_agent_ddpg[i].critic_local.state_dict(),
                           f"agent_{i}_critic_local_checkpoint.pth")
            break

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(score_per_episode)), score_per_episode, c='g', label='Average score per agent')
    plt.plot(np.arange(len(avg_rolling_score)), avg_rolling_score, c='b',
             label='Rolling average over last 100 episodes')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()


def demonstrate_multi_agent(env: UnityEnvironment, agent: MultiAgentDDPG):
    """
    A method to show the agents playing tennis.

    :param env: the unity tennis environment
    :param agent: the ddpg multiagent
    """
    file_path = os.getcwd()
    brain_name = env.brain_names[0]

    agent.multi_agent_ddpg[0].actor_local.load_state_dict(torch.load(
        os.path.join(file_path, "agent_0_actor_local_checkpoint.pth")))
    agent.multi_agent_ddpg[0].critic_local.load_state_dict(torch.load(
        os.path.join(file_path, "agent_0_critic_local_checkpoint.pth")))
    agent.multi_agent_ddpg[1].actor_local.load_state_dict(torch.load(
        os.path.join(file_path, "agent_1_actor_local_checkpoint.pth")))
    agent.multi_agent_ddpg[1].critic_local.load_state_dict(torch.load(
        os.path.join(file_path, "agent_1_critic_local_checkpoint.pth")))

    agent.multi_agent_ddpg[0].actor_local.eval()
    agent.multi_agent_ddpg[0].critic_local.eval()
    agent.multi_agent_ddpg[1].actor_local.eval()
    agent.multi_agent_ddpg[1].critic_local.eval()

    for _ in range(5):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        done = False
        while not np.any(done):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            done = env_info.local_done
            state = next_state
            time.sleep(0.005)


if __name__ == "__main__":
    cur_dir = os.getcwd()
    loc = 'Tennis_Windows_x86_64\\Tennis.exe'
    tennis_env = UnityEnvironment(file_name=os.path.join(cur_dir, loc))

    multi_agent = MultiAgentDDPG(state_size=24, action_size=2, lr_actor=1e-5, lr_critic=1e-4,
                                 replay_buffer_size=int(1e6), gamma=0.95, batch_size=64, random_seed=10,
                                 soft_update_tau=1e-3)

    # train_maddpg(env=tennis_env, agent=multi_agent)

    demonstrate_multi_agent(env=tennis_env, agent=multi_agent)

    tennis_env.close()
