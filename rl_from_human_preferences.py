import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from a2c_agent import A2CAgent
from d_space import DSpace
from reward_function import HumanFeedBackRewardFunction
from train_human_rewarder import HumanRewarderTraining


class RLFromHumanPreferences:
    def __init__(
        self,
        input_shape,
        seed,
        device,
        gamma,
        alpha,
        beta,
        update_every,
        actor,
        critic,
        real_human_check=False,
    ):
        self.real_human_check = real_human_check

        # We create our environment
        self.env = gym.make("Pong-v4", mode=1, obs_type="grayscale")
        self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
        self.env = FrameStack(self.env, 4)

        ACTION_SIZE = self.env.action_space.n
        # We create our agent
        self.agent = A2CAgent(
            input_shape=input_shape,
            action_size=ACTION_SIZE,
            seed=seed,
            device=device,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            update_every=update_every,
            actor_m=actor,
            critic_m=critic,
        )

        # We initialize our D space
        self.class_d_space = DSpace(
            env=self.env, agent=self.agent, real_human_check=real_human_check
        )
        self.D_space = self.class_d_space.D_space

        # We initialize our Human Rewarder
        self.human_rewarder = HumanFeedBackRewardFunction(device)

    def train_from_human_preferences(self, n_episodes=1000):
        # We create our Human Rewarder training class
        train_human_rewarder = HumanRewarderTraining(self.D_space, self.human_rewarder)
        # We pretrain our Human Rewarder
        self.human_rewarder = train_human_rewarder.pretrain_human_rewarder()

        T = 1  # Timestep over all the episodes
        timestep = 0  # timestep over one episoded
        labelled_number = 0  # Number of time we've asked the human to label 2 segments

        scores = []
        scores_fake = []
        T_to_plot = []

        obs_list = []
        real_reward_list = []

        for i_episode in range(0, n_episodes + 1):
            obs, _ = self.env.reset()
            obs = np.array(obs._frames)
            obs = np.ascontiguousarray(obs, dtype=np.float32) / 255

            score = 0
            score_fake = 0
            timestep = 0

            while True:
                # Each 10 timestep we train our HumanRewarder
                if timestep % 10 == 0:
                    self.human_rewarder = train_human_rewarder.train_human_rewarder()

                # We take an action and our agent learn from it
                action, log_prob, entropy = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                fake_reward = self.human_rewarder(obs)
                score_fake += torch.Tensor(fake_reward).mean().cpu().detach().numpy()
                score += reward

                next_obs = np.array(next_obs._frames)
                next_obs = np.ascontiguousarray(next_obs, dtype=np.float32) / 255

                # Each 5 timestep we train our Agent
                self.agent.step(
                    obs,
                    log_prob,
                    entropy,
                    fake_reward[0].mean().cpu().detach().numpy(),
                    terminated or truncated,
                    next_obs,
                )

                obs = next_obs

                ####################
                # Part feed D_space

                obs_list.append(obs[-1])
                real_reward_list.append(reward)
                if len(obs_list) > 50:
                    obs_list.pop(0)
                    real_reward_list.pop(0)

                # We do T*4 because each timestep is composed of 4 frames ? Not sure
                if (1 / 2) ** (((T * 1e3) + 5e6) / 5e6) > labelled_number:
                    if len(obs_list) == 50:
                        print("Feed D_space")
                        self.D_space = self.class_d_space.feeding_d_space(
                            obs_list[0:25],
                            obs_list[25:50],
                            real_reward_list[0:25],
                            real_reward_list[25:50],
                        )
                        obs_list = []
                        real_reward_list = []
                        labelled_number += 1

                timestep += 1
                T += 1

                if terminated or truncated:
                    break

            scores.append(score)  # save most recent score
            scores_fake.append(score_fake)
            T_to_plot.append(T)

            clear_output(True)

            plt.title("Pong")
            plt.plot(T_to_plot, scores, label="RL")
            plt.plot(T_to_plot, scores_fake, label="3k synthetic labels")
            plt.ylabel("Reward")
            plt.xlabel("Timestep")
            plt.legend(loc=4)
            plt.show()

            print("Episode:", i_episode)
        return scores

    def save_model(self):
        torch.save(self.human_rewarder.net.state_dict(), "./Human_rewarder_saved_model")
        self.human_rewarder.net.load_state_dict(
            torch.load("Human_rewarder_saved_model")
        )
