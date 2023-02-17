from random import sample
import torch
import numpy as np
from reward_function import HumanFeedBackRewardFunction


class HumanRewarderTraining:
    def __init__(self, D_space, human_rewarder):
        self.human_rewarder = human_rewarder
        self.D_space = D_space

    def train_human_rewarder(self):
        i = 10  # We sample 10 triplets from D

        D_sampled = sample(self.D_space, i)

        fake_reward_1_list_all = torch.Tensor(
            []
        )  # (i, 3) --> i segments and 3 rewarder
        fake_reward_2_list_all = torch.Tensor([])

        human_choice_list = []

        for triplet in D_sampled:
            fake_reward_1_list = []  # (1, 3) --> 1 segment duos and 3 rewarder
            fake_reward_2_list = []

            human_choice_list.append(triplet[2])

            for n in range(0, len(triplet[0]) - 1, 4):
                obs_1_i = np.array(triplet[0][n : n + 4])
                obs_2_i = np.array(triplet[1][n : n + 4])

                fake_reward_1_list.append(
                    self.human_rewarder(obs_1_i, True)
                )  # We append (1,3)
                fake_reward_2_list.append(self.human_rewarder(obs_2_i, True))

            fake_reward_1_list = torch.sum(
                fake_reward_1_list, axis=0
            )  # From (6,3) to (1, 3)
            fake_reward_2_list = torch.sum(fake_reward_2_list, axis=0)

            if len(fake_reward_1_list_all) == 0:
                fake_reward_1_list_all = fake_reward_1_list.unsqueeze(0)
                fake_reward_2_list_all = fake_reward_2_list.unsqueeze(0)
            else:
                fake_reward_1_list_all = torch.cat(
                    (fake_reward_1_list_all, fake_reward_1_list.unsqueeze(0)), 0
                )
                fake_reward_2_list_all = torch.cat(
                    (fake_reward_2_list_all, fake_reward_2_list.unsqueeze(0)), 0
                )

        human_choice_list = torch.Tensor(human_choice_list)
        fake_reward_1_list_all = torch.Tensor(fake_reward_1_list_all)
        fake_reward_2_list_all = torch.Tensor(fake_reward_2_list_all)

        # Update rewarders weights
        self.human_rewarder.update(
            fake_reward_1_list_all, fake_reward_2_list_all, human_choice_list
        )
        return self.human_rewarder

    def pretrain_human_rewarder(self):
        """In the Atari domain we also pretrain the reward predictor for 200 epochs
        before beginning RL training, to reduce the likelihood of irreversibly
        learning a bad policy based on an untrained predictor."""

        print("Start of the pretrain of the Human Rewarder...")
        for epoch in range(0, 200):
            self.human_rewarder = self.train_human_rewarder()

            if epoch % 20 == 0:
                print("Pretrain of the HumanRewarder, Step:", epoch, "/200")

        print("End of the pretrain of the Human Rewarder.")
        return self.human_rewarder
