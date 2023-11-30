from moo.targets import minimize_deviations, minimize_emissions, minimize_uncertainties


def test_minimize_deviations():
    target = [3270, 3206, 3498]
    target_params = {
        'target_schedule': target}
    max_target_deviation = []

    for time_interval in target:
        max_target_deviation.append(time_interval)
    target_params['max_target_deviation'] = max_target_deviation

    # cluster schedule i 0, maximum deviation is reached, outcome is 1
    cs = [[0, 0, 0]]
    assert minimize_deviations(cs, target_params, schedules=None) == 1
    # cluster schedule equals target, no deviation exists, outcome is 0
    cs = [target]
    assert minimize_deviations(cs, target_params, schedules=None) == 0
    # cluster schedule is half the target, half of the maximum deviation exists
    cs = [i / 2 for i in target]
    assert minimize_deviations(cs, target_params, schedules=None) == 0.5


def test_minimize_emissions():
    target_params = {'emissions': {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                   350: [16, 17, 18, 21, 22, 29, 30],
                                   490: [19, 20, 23, 24, 25, 26, 27, 28]}}
    schedules = {'3': [10, 10, 10]}
    cs = [[10, 10, 10]]
    # schedule id not in emissions list, outcome is 0
    assert minimize_emissions(cs, target_params, schedules) == 0

    schedules = {'19': [10, 10, 10]}
    # schedule id is in list with emissions and schedule values make up complete cs, outcome is maximum (1)
    assert minimize_emissions(cs, target_params, schedules) == 1.0

    cs = [[20, 20, 20]]
    schedules = {'19': [10, 10, 10]}
    # schedule id is in list with emissions, schedule value make up half cs, outcome is 0.5
    assert minimize_emissions(cs, target_params, schedules) == 0.5


def test_minimize_uncertainties():
    target_params = {'uncertainties': {'0': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                       '1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}}
    schedules = {'18': [10, 10, 10]}
    cs = [[10, 10, 10]]
    # schedule id not in uncertainty list, outcome is 0
    assert minimize_uncertainties(cs, target_params, schedules) == 0

    schedules = {'2': [10, 10, 10]}
    cs = [[10, 10, 10]]
    # schedule id in uncertainty list, values make up complete cs, outcome is 1
    assert minimize_uncertainties(cs, target_params, schedules) == 1

    schedules = {'2': [10, 10, 10]}
    cs = [[20, 20, 20]]
    # schedule id in uncertainty list, values make up half cs, outcome is 1
    assert minimize_uncertainties(cs, target_params, schedules) == 0.5
