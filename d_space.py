from get_human_choice import get_human_choice
import numpy as np


class DSpace:
    def __init__(self, env, agent, real_human_check=False):
        self.real_human_check = real_human_check
        print("initialize_D_space: Start")

        self.D_space = []

        while len(self.D_space) < 500:
            obs, _ = env.reset()
            time_step = 0
            obs_1_list = []
            obs_2_list = []
            real_reward_1 = 0
            real_reward_2 = 0

            obs = np.array(obs._frames)
            obs = np.ascontiguousarray(obs, dtype=np.float32) / 255

            while len(self.D_space) < 500:
                time_step += 1

                action, _, _ = agent.act(obs)
                next_obs, reward, _, _, _ = env.step(action)

                next_obs = np.array(next_obs._frames)
                next_obs = np.ascontiguousarray(next_obs, dtype=np.float32) / 255

                obs = next_obs

                if time_step <= 25:
                    obs_1_list.append(
                        obs[-1]
                    )  # We keep only the last frame since 1 step = 1 frame
                    real_reward_1 += reward
                else:
                    obs_2_list.append(
                        obs[-1]
                    )  # We keep only the last frame since 1 step = 1 frame
                    real_reward_2 += reward

                if time_step >= 50:
                    if real_human_check:
                        human_choice = get_human_choice(obs_1_list, obs_2_list)
                    else:
                        # Here we fake the behavior of our real human assessor
                        if real_reward_1 > real_reward_2:
                            human_choice = [1, 0]
                        else:
                            human_choice = [0, 1]

                    if human_choice != [0, 0]:
                        self.D_space.append(
                            np.array([obs_1_list, obs_2_list, human_choice])
                        )

                    obs_1_list = []
                    obs_2_list = []
                    time_step = 0

        print("initialize_D_space: End")

        return self.D_space

    # We shall call this function each T+5e6/5e6 timesteps
    def feeding_d_space(self, obs1, obs2, real_reward_1=[], real_reward_2=[]):
        if self.real_human_check:
            human_choice = get_human_choice(obs1, obs2)
        else:
            # Here we fake the behavior of our real human assessor
            if np.array(real_reward_1).sum() > np.array(real_reward_2).sum():
                human_choice = [1, 0]
            else:
                human_choice = [0, 1]

        if human_choice != [0, 0]:
            self.D_space.append(np.array([obs1, obs2, human_choice]))

        # We keep only the 3000 lastest triplets
        if len(self.D_space) > 3000:
            self.D_space.pop(0)

        return self.D_space
