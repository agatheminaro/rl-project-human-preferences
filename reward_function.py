import torch.nn as nn
import torch.nn.functional as F
import torch


class HumanFeedBackRewardFunction(nn.Module):
    def __init__(self, device):
        """Page 15: (Atari)
        For the reward predictor, we use 84x84 images as inputs (the same as the
        inputs to the policy), and stack 4 frames for a total 84x84x4 input tensor.
        This input is fed through 4 convolutional layers of size 7x7, 5x5, 3x3,
        and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU
        nonlinearities (alpha = 0.01). This is followed by a fully connected layer of
        size 64 and then a scalar output. All convolutional layers use batch norm
        and dropout with alpha = 0.5 to prevent predictor overfitting
        """
        super().__init__()

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.device = device

        self.nets = []
        self.optimizers = []

        self.loss_list = []

        self.__create_rewarders__()

    def __build_network__(self):
        net = nn.Sequential(
            nn.Conv2d(4, 16, (7, 7), 3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (5, 5), 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (3, 3), 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (3, 3), 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.Linear(64, 1),
        )

        return net

    def get_rewarders(self):
        return self.nets

    def __create_rewarders__(self):
        # "Except where otherwise stated we use an ensemble of 3 predictors"
        for i in range(0, 3):
            net = self.__build_network__().to(self.device)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.learning_rate, weight_decay=1e-5
            )

            self.nets.append(net)
            self.optimizers.append(optimizer)

    def forward(self, obs, train=False):
        obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        rewards = []

        for net in self.nets:
            if train:
                net.train()
                rewards.append(net(obs))
            else:
                net.eval()
                with torch.no_grad():
                    rewards.append(net(obs))

        
        return rewards

    def update(self, fake_reward_1, fake_reward_2, human_choice_list):
        """Updates the reward network's weights."""
        loss = 0

        for i in range(0, fake_reward_1.shape[0]):
            loss = 0
            for r_1, r_2, human_choice in zip(
                fake_reward_1[i], fake_reward_2[i], human_choice_list
            ):
                r_1 = r_1.to(self.device)
                r_2 = r_2.to(self.device)
                human_choice = human_choice.to(self.device)

                p_1 = torch.exp(r_1) / (torch.exp(r_1) + torch.exp(r_2))
                p_2 = torch.exp(r_2) / (torch.exp(r_1) + torch.exp(r_2))

                loss -= human_choice[0] * p_1 + human_choice[1] * p_2

            loss = torch.autograd.Variable(loss, requires_grad=True).to(self.device)

        # Update the policy network
        self.optimizers[0].zero_grad()
        self.optimizers[1].zero_grad()
        self.optimizers[2].zero_grad()

        loss.backward()
        self.optimizers[0].step()
        self.optimizers[1].step()
        self.optimizers[2].step()
