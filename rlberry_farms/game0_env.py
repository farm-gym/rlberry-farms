import rlberry.spaces as spaces
from rlberry.envs.interface import Model
import rlberry_farms.farm0.farm as cb
import numpy as np

# Remark there are some stray 1 rewards sometimes I don't know why


class Farm0(Model):
    """
    Farm0 is a very basic 1x1 farm with only one possible plant : beans, planted in a clay ground. The only possible actions are to water the field or to harvest it.
    The observation of the field state is free.
    The advised maximum episode length is 365 (as in 365 days).

    The Farm has the weather of Montpellier in France (e.g. fairly warm weather, well suited for the culture of beans), the initial day is 120. Initially the field is healthy and contains all the nutrient necessary to the plant.

    The reward is the number of grams of harvested beans.

    The condition for end of episode (self.step returns done) is that the day is >= 365 or that the field has been harvested, or that the plant is dead.

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
    
    Actions:
        The action is either watering the field with 1L to 5L of water, harvesting or doing nothing.
    """
    name = "Farm0"
    def __init__(self):
        # init base classes
        Model.__init__(self)

        self.farm = cb.env()
        self.farm.monitor = None
        # observation and action spaces
        # Day, temp mean, temp min, temp max, rain amount, sun exposure, consecutive dry day, stage, size#cm
        high = np.array([365, 50, 50, 50, 300, 5, 100, 10, 200 ])
        low =  np.array([0, -50, -50, -50, 0,   0, 0, 0, 0])
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(7)

        # initialize
        self.state = None
        self.reset()

    def reset(self):
        observation = self.farm.gym_reset()
        return self.farmgymobs_to_obs(observation)

    def step(self, action):
        obs1, _, _, info = self.farm.farmgym_step([])
        obs, reward, is_done, info = self.farm.farmgym_step(self.num_to_action(action))
        return self.farmgymobs_to_obs([obs1[i][5] for i in range(len(obs1))]), reward, is_done, info

    def farmgymobs_to_obs(self, obs):
        return np.array([ float(obs[0]),
                          float(obs[1]['mean#°C'][0]),
                          float(obs[1]['min#°C'][0]),
                          float(obs[1]['max#°C'][0]),
                          float(obs[2]),
                          float(obs[3]),
                          float(obs[4][0]),
                          float(obs[5][0][0]),
                          float(obs[6][0][0][0]),
                         ]
                        )

    def num_to_action(self, num):
        if (num >= 1) and (num <= 5) :
            return [('BasicFarmer-0', 'Field-0', 'Soil-0', 'water_discrete', {'plot': (0, 0), 'amount#L': num, 'duration#min': 60})]
        elif num == 6:
            return [('BasicFarmer-0', 'Field-0', 'Plant-0', 'harvest', {})]
        else:
            return [] # Do nothing.


