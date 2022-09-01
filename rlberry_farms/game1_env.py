import rlberry.spaces as spaces
from rlberry.envs.interface import Model
import rlberry_farms.farm1.farm as cb
import numpy as np


class Farm1(Model):
    """
    TODO : explain more
    Farm1 is a difficult 1x1 farm with only one possible plant : beans, planted in a clay ground.
    The observation of the field state is free.
    The advised maximum episode length is 365 (as in 365 days).

    The Farm has the weather of Montpellier in France (e.g. fairly warm weather, well suited for the culture of beans), the initial day is 120. Initially the field is healthy and contains all the nutrient necessary to the plant.

    The reward is the number of grams of harvested beans, and there is a high negative reward for very low microlife in soil (due to pesticides).

    The condition for end of episode (self.step returns done) is that the day is >= 365 or that the plant is dead.

    Parameters
    ----------

    monitor: boolean, default = True
        If monitor is True, then some (unobserved) variables are saved to a writer that is displayed during training.

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
        - Birds population#nb
        - fertilizer amount#kg
        - Pests plot_population#nb
        - Pollinators occurrence#bin
        - Weeds grow#nb
        - Weeds flowers#nb

    Actions:
        The action is either watering the field with 1L to 5L of water, harvesting or doing nothing, or .... TODO
    """

    name = "Farm0"

    def __init__(self, monitor = True):
        # init base classes
        Model.__init__(self)

        self.farm = cb.env()
        self.farm.monitor = None
        # observation and action spaces
        # Day, temp mean, temp min, temp max, rain amount, sun exposure, consecutive dry day, stage, size#cm, wet surface, microlife %,
        # bird pop, fertilizer amount, pests pop, pollinators occurrence, weeds grow nb, weeds flower nb
        high = np.array(
            [365, 50, 50, 50, 300, 5, 100, 10, 200, 1, 100, 20, 10, 1000, 1, 100, 100]
        )
        low = np.array([0, -50, -50, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(14)

        # monitoring writer
        self.writer = DefaultWriter(name="farm_writer", log_interval = 5)
        self.monitor_variables = self.farm.monitor_variables
        self.iteration = 0
        self.monitor = monitor
        
        # initialize
        self.state = None
        self.reset()

    def reset(self):
        observation = self.farm.gym_reset()
        self.iteration = 0

        return self.farmgymobs_to_obs(observation)

    def step(self, action):
        obs1, _, _, info = self.farm.farmgym_step([])
        obs, reward, is_done, info = self.farm.farmgym_step(self.num_to_action(action))
        if hasattr(reward, "__len__"):
            reward = reward[0]

        # Monitoring
        if self.monitor:
            self.iteration += 1
            for i in range(len(self.monitor_variables)):
                v= self.monitor_variables[i]
                fi_key,entity_key,var_key,map_v,name_to_display, v_range = v
                day = self.farm.fields[fi_key].entities['Weather-0'].variables['day#int365'].value
                value = map_v(self.farm.fields[fi_key].entities[entity_key].variables[var_key])
                self.writer.add_scalar(var_key, np.round(value,3),self.iteration)
            self.writer.add_scalar('day#int365', day, self.iteration)

        if obs1[8][5][0][0][0] < 20:
            reward -= 300  # if microlife is < 20%, negative reward
        if obs1[8][5][0][0][0] < 10:
            reward -= 300  # if microlife is < 10%, negative reward

        observation = self.farmgymobs_to_obs([obs1[i][5] for i in range(len(obs1))])
        return observation, reward, is_done, info

    def farmgymobs_to_obs(self, obs):
        return np.array(
            [
                obs[0],
                obs[1]["mean#°C"][0],
                obs[1]["min#°C"][0],
                obs[1]["max#°C"][0],
            ]
            + [np.array([obs[i]]).ravel()[0] for i in range(2, 15)]
        )

    def num_to_action(self, num):
        if (num >= 1) and (num <= 5):
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Soil-0",
                    "water_discrete",
                    {"plot": (0, 0), "amount#L": num, "duration#min": 60},
                )
            ]
        elif num == 6:
            return [("BasicFarmer-0", "Field-0", "Plant-0", "harvest", {})]
        elif num == 7:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Plant-0",
                    "sow",
                    {"plot": (0, 0), "amount#seed": 1, "spacing#cm": 10},
                )
            ]
        elif num == 8:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Facility-0",
                    "put_scarecrow",
                    {"type": "basic"},
                )
            ]
        elif num == 9:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Facility-0",
                    "put_scarecrow",
                    {"type": "advanced"},
                )
            ]
        elif num == 10:
            return [("BasicFarmer-0", "Field-0", "Facility-0", "remove_scarecrow", {})]
        elif num == 11:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Fertilizer-0",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 12:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Cide-0",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 13:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Cide-1",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 14:
            return [("BasicFarmer-0", "Field-0", "Weeds-0", "remove", {"plot": (0, 0)})]
        else:
            return []  # Do nothing.
