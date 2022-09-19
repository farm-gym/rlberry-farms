import rlberry.spaces as spaces
from rlberry.envs.interface import Model
import rlberry_farms.farm0.farm as cb
from rlberry_farms.utils import (
    farmgymobs_to_obs,
    update_farm_writer,
    observation_hide_final_state_of_plants,
)
import numpy as np
from rlberry.utils.writers import DefaultWriter
import time
import os
from rlberry.utils.check_env import check_env, check_rlberry_env


class Farm0(Model):
    """
    Farm0 is a very basic 1x1 farm with only one possible plant : tomato, planted in a clay ground.
    The actions are to water the field or to harvest it.
    The advised maximum episode length is 365 (as in 365 days).

    The Farm has the weather of Montpellier in France (e.g. fairly warm weather, well suited for the culture of tomato), the initial day is 100. Initially the field is healthy and contains all the nutrient necessary to the plant.

    The reward is the number of grams of harvested tomatoes.

    The condition for end of episode (self.step returns done) is that the day is >= 365 or that the field has been harvested, or that the plant is dead.

    Parameters
    ----------
    monitor: boolean, default = True
        If monitor is True, then some (unobserved) variables are saved to a writer.
        The writer is available as self.writer. In particular, see self.writer.data.
    enable_tensorboard: boolean, default = False
        If True and monitor is True, save writer as tensorboard data
    output_dir: str, default = "results"
        directory where writer data are saved

    Notes
    -----
    State:
        The state consists of
        - Day (from 1 to 365)
        - mean air temperature (°C)
        - min air temperature (°C)
        - max air temperature (°C)
        - rain amount (mm)
        - sun-exposure (from 1 to 5)
        - consecutive dry day (int)
        - stage of growth of the plant (int)
        - size of the plant in cm
        - numbre of fruits (int)
        - weight of fruits (g).

    Actions:
        The actions are :
        - doing nothing.
        - 2 levels of watering the field (1L or 5L of water)
        - harvesting
    """

    name = "Farm0"

    observations_txt = [
        "Day (from 1 to 365)",
        "Mean air temperature (°C)",
        "Min air temperature (°C)",
        "Max air temperature (°C)",
        "Rain amount",
        "Sun-exposure (from 1 to 5)",
        "Consecutive dry day (int)",
        "Stage of growth of the plant",
        "Size of the plant in cm",
        "nb of fruits",
        "weight of fruits",
    ]

    def __init__(self, monitor=False, enable_tensorboard=False, output_dir="results"):
        # init base classes
        Model.__init__(self)

        self.farm = cb.env()
        self.farm.monitor = None
        self.farm.gym_step([])
        # observation and action spaces
        # Day, temp mean, temp min, temp max, rain amount, sun exposure, consecutive dry day, stage, size#cm, fruit weight, nb of fruits, weights
        high = np.array([365, 50, 50, 50, 300, 7, 100, 10, 200, 100, 5000])
        low = np.array([0, -50, -50, -50, 0, 0, 0, 0, 0, 0, 0])
        self.n_obs = len(high)
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(4)

        # monitoring writer
        params = {}
        self.identifier = self.name + str(self.seeder.rng.integers(100000))
        self.output_dir = output_dir
        if enable_tensorboard:
            self.tensorboard_dir = os.path.join(output_dir, "tensorboard")
            params["tensorboard_kwargs"] = dict(
                log_dir=os.path.join(self.tensorboard_dir, "farm_" + self.identifier)
            )
        self.writer = DefaultWriter(name="farm_writer", print_log=False, **params)
        self.monitor_variables = self.farm.monitor_variables
        self.iteration = 0
        self.monitor = monitor
        if self.monitor:
            keys = []
            for i in range(len(self.monitor_variables)):
                v = self.monitor_variables[i]
                fi_key, entity_key, var_key, map_v, name_to_display, v_range = v
                keys.append(var_key)
            layout = {
                "Farm": {
                    "Variables": ["Multiline", keys],
                },
            }
            self.writer.add_custom_scalars(layout)

        # initialize
        self.state = None
        self.reset()

    def reset(self):
        observation = self.farm.gym_reset()
        obs1, _, _, info = self.farm.gym_step([])
        return observation_hide_final_state_of_plants(
            farmgymobs_to_obs(obs1), id_of_plants_stage=7
        )

    def writer_to_csv(self):
        self.writer.data.to_csv(
            os.path.join(self.output_dir, "farm_" + self.identifier + "_writer.csv")
        )

    def step(self, action):
        # Stepping
        #   farmgym run with a cycle of 2 steps: 1 (empty) step of getting observation ("morning"), then 1 step of acting ("afternoon").
        #   Classic RL methodology use only 1 step : performing an action return the next observation
        #   To match this 2, rlberry_farms run the 'farmgy observation step ("morning")' right after the action.
        #   With this method, it will be like classic RL 'step' for the user
        _, reward, is_done, info = self.farm.farmgym_step(self.num_to_action(action))
        obs1, _, _, info = self.farm.gym_step([])

        # Monitoring
        if self.monitor:
            self.iteration += 1
            update_farm_writer(
                self.writer, self.monitor_variables, self.farm, self.iteration
            )

        return (
            observation_hide_final_state_of_plants(
                farmgymobs_to_obs(obs1), id_of_plants_stage=7
            ),
            reward,
            is_done,
            info,
        )

    def num_to_action(self, num):
        if num == 1:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Soil-0",
                    "water_discrete",
                    {"plot": (0, 0), "amount#L": 1, "duration#min": 60},
                )
            ]
        elif num == 2:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Soil-0",
                    "water_discrete",
                    {"plot": (0, 0), "amount#L": 5, "duration#min": 60},
                )
            ]
        elif num == 3:
            return [("BasicFarmer-0", "Field-0", "Plant-0", "harvest", {})]
        else:
            return []  # Do nothing.


if __name__ == "__main__":
    check_env(Farm0())
    check_rlberry_env(Farm0())
