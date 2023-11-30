import numpy as np

from mango_library.negotiation.multiobjective_cohda.data_classes import Target


def minimize_deviations(cs, target_params, schedules):
    """
    calculate for each time interval the deviation from target schedule and normalize sum between 0 and 1
    zi = (xi – min(x)) / (max(x) – min(x))
    :param cs: cluster schedule
    :param target_params: needs target schedule and max target deviation
    :param schedules: not required
    :return: normalized sum of all deviations
    """
    sum_cs = np.sum(cs, axis=0)
    diff = np.abs(target_params["target_schedule"] - sum_cs)  # deviation to the target schedule
    result = np.sum(diff)
    total_max_target_deviation = np.sum(target_params["max_target_deviation"])
    norm_result = (result - 0) / (total_max_target_deviation - 0)

    return norm_result


def minimize_emissions(cs, target_params, schedules):
    """
    Calculates how much of the total produced power would cause emissions,
    using relative values.
    :param cs: cluster schedule
    :param target_params: needs emission values and max_emissions
    :param schedules: individual schedules from all agents for the solution point to evaluate
    :return: normalized sum of all emissions over all time intervals
    """
    # total produced power
    sum_cs = np.sum(cs, axis=0)
    # produced power with emissions
    emissions_cs = []
    for emissions, idx_list in target_params["emissions"].items():
        if int(emissions) != 0:
            for idx in idx_list:
                if str(idx - 1) in schedules:
                    emissions_cs.append(schedules[str(idx - 1)])

    all_emissions = 0

    if emissions_cs:
        sum_emissions_cs = np.sum(emissions_cs, axis=0)

        for time_interval, emissions_gen in enumerate(sum_emissions_cs):
            # calculate how much of total produced power in time interval produces emissions
            if sum_cs[time_interval] == 0:
                emissions = 0
            else:
                emissions = emissions_gen / sum_cs[time_interval]

            all_emissions += emissions

    norm_all_emissions = (all_emissions - 0) / (1 * len(sum_cs) - 0)
    return norm_all_emissions


def minimize_uncertainties(cs, target_params, schedules):
    """
    Calculates how much of the total produced power is uncertain. Units can be fully uncertain or certain.
    If all power is produced by certain units, uncertainty value is 0. Same if no power is produced at all.
    Uncertainty increases linear over time (=first time interval has smallest, last time interval largest uncertainty)
    Attention: Uncertainty value does NOT depend on total amount of power - if very less power is in schedule but fully
    produced by uncertain unit it still gets highest uncertainty value
    To keep in mind: currently the total sum of all uncertainty values is always 100, that means with less time
    intervals the uncertainty values increase faster
    :param cs: cluster schedule
    :param target_params: needs list with certain/uncertain units
    :param schedules: individual schedules from all agents for the solution point to evaluate
    :return: normalized uncertainty value
    """
    # total produced power
    sum_cs = np.sum(cs, axis=0)
    # uncertain produced power
    uncertain_cs = []
    for uncertain_idx in target_params['uncertainties']['1']:
        if str(uncertain_idx) in schedules:
            uncertain_cs.append(schedules[str(uncertain_idx)])

    total_uncertainty = 0

    # based on x + 2x + 3x + 4x ... + len(cs[0]) = 100 -> uncertainty values per time interval
    gaussian_sum = (len(cs[0]) * (len(cs[0]) + 1)) / 2
    x = 100 / gaussian_sum

    if uncertain_cs:
        uncertain_aggregated_schedule = np.sum(uncertain_cs, axis=0)
        for time_interval, uncertain_gen in enumerate(uncertain_aggregated_schedule):
            # calculate how much of total produced power in time interval is uncertain (between 0 and 1.0)
            if sum_cs[time_interval] == 0:
                perc_of_total = 0
            else:
                perc_of_total = uncertain_gen / sum_cs[time_interval]

            if perc_of_total == 0:
                uncertainty = 0
            else:
                # get weight for uncertainty value depending on time interval and calculate uncertainty
                weight = time_interval + 1 * x
                uncertainty = weight * perc_of_total
            total_uncertainty += uncertainty

    # calculate max possible uncertainty: In every time interval the whole power is uncertain
    max_uncertainty = 0
    for i in range(len(cs[0])):
        max_uncertainty += (i + 1 * x)

    norm_total_uncertainty = (total_uncertainty - 0) / (max_uncertainty - 0)
    return norm_total_uncertainty


TARGETS = [Target(target_function=minimize_deviations, ref_point=1.1, maximize=False),
           Target(target_function=minimize_emissions, ref_point=1.1, maximize=False),
           Target(target_function=minimize_uncertainties, ref_point=1.1, maximize=False)
           ]
