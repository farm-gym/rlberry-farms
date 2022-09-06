import numpy as np


def farmgymobs_to_obs(obs_lst):
    """
    Exctract all the numerical values of a farmgym obs and return a flat array.
    """
    res = np.array([])
    for obs in obs_lst:
        if type(obs) is dict:
            to_add = np.array(list(obs.values()))
        else:
            to_add = np.array([obs])

        res = np.concatenate((res, to_add), axis=None)
    return res


def update_farm_writer(writer, monitor_variables, farm, iteration):
    """
    Update the farm writer with the valued monitored by the farm.
    """
    for i in range(len(monitor_variables)):
        v = monitor_variables[i]
        fi_key, entity_key, var_key, map_v, name_to_display, v_range = v
        day = farm.fields[fi_key].entities["Weather-0"].variables["day#int365"].value
        value = map_v(farm.fields[fi_key].entities[entity_key].variables[var_key])
        writer.add_scalar(var_key, np.round(value, 3), iteration)
    writer.add_scalar("day#int365", day, iteration)
