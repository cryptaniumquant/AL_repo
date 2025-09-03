def cast_params_to_types(experiment_params):

    for param_name in experiment_params:
        if experiment_params[param_name]["type"] == "int":
            experiment_params[param_name]["v"] = int(experiment_params[param_name]["v"])
        elif experiment_params[param_name]["type"] == "float":
            experiment_params[param_name]["v"] = float(experiment_params[param_name]["v"])
            
def params_fit_in_intervals(experiment_params) -> bool:
    count = 0
    for param_name in experiment_params:
        if experiment_params[param_name]["min"] < experiment_params[param_name]["v"] and experiment_params[param_name]["v"] < experiment_params[param_name]["max"]:
            count += 1

    if count == len(experiment_params):
        return True
    else:
        return False
        