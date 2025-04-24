import json
import numpy as np

def willingness_to_cycle(tij, location, mode="fiets", param_path=None):
    """
    Calculate the willingness to cycle based on travel time and location.

    Args:
        tij (float, array): Travel time in minutes.
        location (str): Location code.
        mode (str): One of "fiets" or "ebike. Default is "fiets".
    """ 
    if param_path is None:
        param_path = f"data/cycle_willingness/{mode}.json"
    
    with open(param_path, "r") as f:
        model_params = json.load(f)

    if location not in model_params:
        raise ValueError("Unknown location")
    
    a, b = model_params[location]
    F_ij = 1 / (1 + np.exp(a + b * np.log(tij + 1e-10)))
    
    return F_ij