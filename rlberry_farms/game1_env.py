import rlberry.spaces as spaces
from rlberry.envs.interface import Model
import rlberry_farms.farm1.farm as cb
import numpy as np
import time
import os
from rlberry.utils.writers import DefaultWriter
from rlberry_farms.utils import (
    farmgymobs_to_obs,
    update_farm_writer,
    observation_hide_final_state_of_plants,
)

class Farm1(Model):
    """
    Farm1 is a difficult 1x1 farm with only one possible plant : beans, planted in a clay ground.
    The advised maximum episode length is 365 (as in 365 days).

    The Farm has the weather of Lille in France (e.g. well suited for the culture of beans), the initial day is 120. Initially the field is healthy and contains all the nutrient necessary to the plant.

    The reward is the number of grams of harvested beans, and there is a negative reward for very low microlife in soil (due to pesticides).

    The condition for end of episode (self.step returns done) is that the day is >= 365 or that the plant is dead.

    Parameters
    ----------
    monitor: boolean, default = True
        If monitor is True, then some (unobserved) variables are saved to a writer that is displayed during training.
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
        - size of the plant in cm.
        - Soil wet_surface#m2.day-1
        - fertilizer amount#kg
        - Pests plot_population#nb
        - Pollinators occurrence#bin
        - Weeds grow#nb
        - Weeds flowers#nb
        - Weight of fruits (g)
        - Microlife health index (%)

    Actions:
        The actions are :
        - doing nothing.
        - 2 levels of watering the field (1L or 5L of water)
        - harvesting
        - sow some seeds
        - scatter fertilizer
        - scatter herbicide
        - scatter pesticide
        - remove weeds by hand
    """

    name = "Farm1"

    observations_txt = [
        "Day (from 1 to 365)",
        "Mean air temperature (°C)",
        "Min air temperature (°C)",
        "Max air temperature (°C)",
        "Rain amount",
        "Sun-exposure (from 1 to 5)",
        "Consecutive dry day (int)",
        "Stage of growth of the plant",
        "Number of fruits (int)",
        "Size of the plant in cm",
        "Soil wet_surface (m2.day-1)",
        "fertilizer amount (kg)",
        "Pollinators occurrence (bin)",
        "Weeds grow (nb)",
        "Weeds flowers (nb)",
        "weight of fruits",
        "microlife health index (%)",
    ]

    def __init__(self, monitor=True, enable_tensorboard=False, output_dir="results"):
        # init base classes
        Model.__init__(self)

        self.farm = cb.env()
        self.farm.gym_step([])

        self.farm.monitor = None
        # observation and action spaces
        # Day, temp mean, temp min, temp max, rain amount, sun exposure, consecutive dry day, stage, size#cm, nb of fruits,
        # wet surface,  fertilizer amount,  pollinators occurrence, weeds grow nb, weeds flower nb, weight of fruits, microlife health index %
        high = np.array(
            [365, 50, 50, 50, 300, 5, 10, 7, 100, 300, 10, 10, 1, 100, 100, 5000, 100],dtype=np.float64
        )
        low = np.array([0, -50, -50, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],dtype=np.float64)
        self.observation_space = spaces.Box(low=low, high=high,dtype=np.float64)
        self.action_space = spaces.Discrete(9)

        # monitoring writer
        self.identifier = self.name + str(self.seeder.rng.integers(100000))
        params = {}
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

        # initialize
        self.state = None
        self.reset()

    def reset(self):
        observation = self.farm.gym_reset()
        self.farm.gym_step([])
        return observation_hide_final_state_of_plants(
            farmgymobs_to_obs(observation), id_of_plants_stage=7
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

        if hasattr(reward, "__len__"):
            reward = reward[0]

        # Monitoring
        if self.monitor:
            self.iteration += 1
            update_farm_writer(
                self.writer, self.monitor_variables, self.farm, self.iteration
            )
        if np.array(obs1[-1]).item() < 10:
            reward -= 2  # if microlife is < 10%, negative reward

        observation = observation_hide_final_state_of_plants(
            farmgymobs_to_obs(obs1), id_of_plants_stage=7
        )
        return observation, reward, is_done, info

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
        elif num == 4:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Plant-0",
                    "sow",
                    {"plot": (0, 0), "amount#seed": 1, "spacing#cm": 10},
                )
            ]
        elif num == 5:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Fertilizer-0",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 6:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Cide-0",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 7:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Cide-1",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 8:
            return [("BasicFarmer-0", "Field-0", "Weeds-0", "remove", {"plot": (0, 0)})]
        else:
            return []  # Do nothing.
