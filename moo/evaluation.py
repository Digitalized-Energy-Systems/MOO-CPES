import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from evoalgos.selection import HyperVolumeContributionSelection

from moo.config import DB_FILE, NUM_SIMULATIONS, NUM_SOLUTION_POINTS, NUM_AGENTS
from config import TARGET
from scipy.stats import sem
import matplotlib.lines as mlines

class Individual:

    def __init__(self, objective_values):
        self._objective_values = objective_values

    @property
    def objective_values(self):
        return self._objective_values

    @objective_values.setter
    def objective_values(self, objective_values):
        self._objective_values = objective_values


def create_reference_front(paths_to_reference_front):
    selection = HyperVolumeContributionSelection(prefer_boundary_points=False)
    selection.sorting_component.hypervolume_indicator.reference_point = [1.1, 1.1, 1.1]

    population = []

    for path, number_of_runs in paths_to_reference_front.items():
        with h5py.File(path, 'r') as results:
            for iter in range(number_of_runs):
                current_iter = results['iteration ' + str(0)]
                performances = np.array(current_iter['performances'])
                for performance_tuple in performances:
                    if isinstance(performance_tuple[0], float):
                        individual = Individual(performance_tuple)
                    else:
                        individual = Individual((performance_tuple[0][0], performance_tuple[1][0], performance_tuple[2][0]))
                    population.append(individual)
                    # f1 = float(performance_tuple[0])
                    # f2 = float(performance_tuple[1])
                    # f3 = float(performance_tuple[2])
                    # merged_approximated_front.append([f1, f2, f3])

    # find the non-dominated elements
    min_elements = selection.sorting_component.non_dom_sorting.compute_non_dom_front_arbitrary_dim(population)
    hv_ref = selection.sorting_component.hypervolume_indicator.assess_non_dom_front(min_elements)
    print("HV ref front", hv_ref)

    with h5py.File('ref_front.hdf5', "w") as f:
        grp = f.create_group(f"front")
        for idx in range(len(min_elements)):
            print(idx, ":", min_elements[idx].objective_values)
            grp.create_dataset(f"entries{idx}", data=min_elements[idx].objective_values)
        grp.create_dataset(f"hypervolume", data=hv_ref)


def target_fulfilled(cluster_schedule):
    return all(round(x) == round(y) for x, y in zip(cluster_schedule, TARGET))


def sort_performances(f1, f2):
    tuple_list = [(a, b) for a, b in zip(f1, f2)]
    tuple_list.sort(key=lambda x: x[0])
    return tuple_list


def plot_single_fronts(merged_approximated_front, path_to_reference_front):
    reference_front = get_reference_front(path=path_to_reference_front)
    ref_front_f1 = np.array(reference_front)[:, 0]
    ref_front_f2 = np.array(reference_front)[:, 1]
    ref_front_f3 = np.array(reference_front)[:, 2]

    f1 = np.array(merged_approximated_front)[:, 0]
    f2 = np.array(merged_approximated_front)[:, 1]
    f3 = np.array(merged_approximated_front)[:, 2]

    # f1, f2
    fig, ax = plt.subplots()
    ax.scatter(f1,
               f2, c='blue')

    plot_front = sort_performances(ref_front_f1, ref_front_f2)
    plot_front_1 = [x[0] for x in plot_front]
    plot_front_2 = [x[1] for x in plot_front]
    line = mlines.Line2D(plot_front_1, plot_front_2, color='red')
    line.set_linewidth(3)
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.set_xlabel('minimize deviations', labelpad=20)
    ax.set_ylabel('minimize emissions', labelpad=20)
    plt.legend([line], ['Pareto Frontier'])
    plt.show()

    # f1, f3
    fig, ax = plt.subplots()
    ax.scatter(f1, f3,
               c='blue')
    plot_front = sort_performances(ref_front_f1, ref_front_f3)
    plot_front_1 = [x[0] for x in plot_front]
    plot_front_2 = [x[1] for x in plot_front]
    line = mlines.Line2D(plot_front_1, plot_front_2, color='red')
    line.set_linewidth(3)
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.set_xlabel('minimize deviations', labelpad=20)
    ax.set_ylabel('minimize uncertainties', labelpad=20)
    plt.legend([line], ['Pareto Frontier'])
    plt.show()

    # f2, f3
    fig, ax = plt.subplots()
    ax.scatter(f2, f3, color='blue')
    plot_front = sort_performances(ref_front_f2, ref_front_f3)
    plot_front_1 = [x[0] for x in plot_front]
    plot_front_2 = [x[1] for x in plot_front]
    line = mlines.Line2D(plot_front_1, plot_front_2, color='red')
    line.set_linewidth(3)
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.set_xlabel('minimize emissions', labelpad=20)
    ax.set_ylabel('minimize uncertainties', labelpad=20)
    plt.legend([line], ['Pareto Frontier'])
    plt.show()


def get_reference_front(path):
    merged_approximated_front = []

    with h5py.File(path, 'r') as results:
        size = len(np.array(results['front']))-1
        for idx in range(size):
            # performances = np.array(results['front']['entries' + str(idx)])
            # f1 = float(performances[0])
            # f2 = float(performances[1])
            # f3 = float(performances[2])
            f1 = results['front']['entries' + str(idx)]['Performance_0']
            f2 = results['front']['entries' + str(idx)]['Performance_1']
            f3 = results['front']['entries' + str(idx)]['Performance_2']
            merged_approximated_front.append([f1, f2, f3])

    return merged_approximated_front


def analyze_results(paths, total_number_of_runs, path_to_reference_front):
    average_hv = 0
    merged_approximated_front = []
    all_hv = []
    all_duration = []
    all_messages = []
    average_duration = 0
    average_messages = 0
    all_min_f1 = []
    all_min_f2 = []
    all_min_f3 = []
    average_min_f1 = 0
    average_min_f2 = 0
    average_min_f3 = 0

    merged_front_targets_fulfilled = []
    hv_progression_total = {}
    hv_progression_after_decide = {}
    hv_progression_after_update = {}
    all_agents_known = {}

    for path, number_of_runs in paths.items():
        with h5py.File(path, 'r') as results:
            for r_idx in range(number_of_runs):
                current_iter = results['iteration ' + str(r_idx)]
                performances = np.array(current_iter['performances'])
                run_min_f1 = 1
                run_min_f2 = 1
                run_min_f3 = 1
                for performance_tuple in performances:
                    if isinstance(performance_tuple[0], float):
                        f1 = float(performance_tuple[0])
                        f2 = float(performance_tuple[1])
                        f3 = float(performance_tuple[2])
                    else:
                        f1 = float(performance_tuple[0][0])
                        f2 = float(performance_tuple[1][0])
                        f3 = float(performance_tuple[2][0])
                    merged_approximated_front.append([f1, f2, f3])
                    if f1 < run_min_f1:
                        run_min_f1 = f1
                    if f2 < run_min_f2:
                        run_min_f2 = f2
                    if f3 < run_min_f3:
                        run_min_f3 = f3
                hv = np.float64(current_iter['hypervolume'])
                duration = float(str(np.string_(current_iter['duration']).tobytes())[2:-1])
                messages = np.float64(current_iter['messages'])
                average_hv += hv
                all_hv.append(hv)
                average_duration += duration
                all_duration.append(duration)
                average_messages += messages
                all_messages.append(messages)
                average_min_f1 += run_min_f1
                all_min_f1.append(run_min_f1)
                average_min_f2 += run_min_f2
                all_min_f2.append(run_min_f2)
                average_min_f3 += run_min_f3
                all_min_f3.append(run_min_f3)

                # for s_idx in range(NUM_SOLUTION_POINTS):
                #     cs = np.array(current_iter['cluster schedules'][f'cluster_schedule_{s_idx}'])
                #     if target_fulfilled(np.sum(cs, axis=0)):
                #         merged_front_targets_fulfilled.append([float(performances[s_idx][0]), float(performances[s_idx][1]),
                #                                                float(performances[s_idx][2])])

                # for a_idx in range(NUM_AGENTS):
                #     # TODO check index and order
                #     hvs_keys = np.array(current_iter['hypervolumes per agent total'][f"hv_list_{a_idx}_keys"])
                #     hvs_values = np.array(current_iter['hypervolumes per agent total'][f"hv_list_{a_idx}_values"])
                #     if r_idx not in hv_progression_total:
                #         hv_progression_total[r_idx] = {}
                #     if a_idx not in hv_progression_total[r_idx].keys():
                #         hv_progression_total[r_idx][a_idx] = {}
                #     hv_progression_total[r_idx][a_idx]['time'] = hvs_keys
                #     hv_progression_total[r_idx][a_idx]['hv'] = hvs_values
                #
                #     hvs_keys = np.array(current_iter['hypervolumes per agent after decide'][f"hv_list_{a_idx}_keys"])
                #     hvs_values = np.array(current_iter['hypervolumes per agent after decide'][f"hv_list_{a_idx}_values"])
                #     if r_idx not in hv_progression_after_decide:
                #         hv_progression_after_decide[r_idx] = {}
                #     if a_idx not in hv_progression_after_decide[r_idx].keys():
                #         hv_progression_after_decide[r_idx][a_idx] = {}
                #     hv_progression_after_decide[r_idx][a_idx]['time'] = hvs_keys
                #     hv_progression_after_decide[r_idx][a_idx]['hv'] = hvs_values
                #
                #     hvs_keys = np.array(current_iter['hypervolumes per agent after update'][f"hv_list_{a_idx}_keys"])
                #     hvs_values = np.array(current_iter['hypervolumes per agent after update'][f"hv_list_{a_idx}_values"])
                #     if r_idx not in hv_progression_after_update:
                #         hv_progression_after_update[r_idx] = {}
                #     if a_idx not in hv_progression_after_update[r_idx].keys():
                #         hv_progression_after_update[r_idx][a_idx] = {}
                #     hv_progression_after_update[r_idx][a_idx]['time'] = hvs_keys
                #     hv_progression_after_update[r_idx][a_idx]['hv'] = hvs_values
                #
                #     dates = np.array(current_iter['all agents known'][f"agents_known_agent_{a_idx}"])
                #     if r_idx not in all_agents_known:
                #         all_agents_known[r_idx] = []
                #     all_agents_known[r_idx].append(dates)

    average_hv = average_hv / total_number_of_runs
    average_duration = average_duration / total_number_of_runs
    average_messages = average_messages / total_number_of_runs
    average_min_f1 = average_min_f1 / total_number_of_runs
    average_min_f2 = average_min_f2 / total_number_of_runs
    average_min_f3 = average_min_f3 / total_number_of_runs
    min_hv = min(all_hv)
    max_hv = max(all_hv)
    sem_hv = sem(all_hv)
    sem_duration = sem(all_duration)
    sem_messages = sem(all_messages)
    sem_min_f1 = sem(all_min_f1)
    sem_min_f2 = sem(all_min_f2)
    sem_min_f3 = sem(all_min_f3)
    print(f'Average HV: {average_hv}, SEM HV: {sem_hv}')
    print(f'Average Duration: {average_duration}, SEM Duration: {sem_duration}')
    print(f'Average Messages: {average_messages}, SEM Duration: {sem_messages}')
    print(f'Average min f1: {average_min_f1}, SEM min f1: {sem_min_f1}')
    print(f'Average min f2: {average_min_f2}, SEM min f2: {sem_min_f2}')
    print(f'Average min f3: {average_min_f3}, SEM min f3: {sem_min_f3}')



    # # show only results with fulfilled targets
    # if len(merged_front_targets_fulfilled) == len(merged_approximated_front):
    #     print(f'All results fulfill complete target schedule.')
    # elif len(merged_front_targets_fulfilled) == 0:
    #     print('No points complete fulfill target schedule.')
    # else:
    #     plt.figure(figsize=(10, 10))
    #     ax = plt.axes(projection='3d')
    #     ax.grid()
    #
    #     ax.scatter(np.array(merged_front_targets_fulfilled)[:, 0], np.array(merged_front_targets_fulfilled)[:, 1],
    #                np.array(merged_front_targets_fulfilled)[:, 2], c='blue', s=50)
    #     ax.set_title('Results Energy Scenario, only fulfilled target schedules')
    #
    #     # Set axes label
    #     ax.set_xlabel('minimize deviations', labelpad=20)
    #     ax.set_ylabel('minimize emissions', labelpad=20)
    #     ax.set_zlabel('f3', labelpad=20)
    #     plt.show()

    # plot_single_fronts(merged_approximated_front=merged_approximated_front,
    #                    path_to_reference_front=path_to_reference_front)

    plot_two_objective_fronts(merged_approximated_front)

    # only plot hv progression for one iteration
    iteration = 0
    time_all_agents_known = max(all_agents_known[iteration])
    print(f'Time for complete knowledge of all agents: {time_all_agents_known}')

    colors_complete = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors_complete.items())
    colors = [x[1] for x in by_hsv]

    # plot hv progression in total per agent
    fig, ax = plt.subplots()
    current_hv_progression = hv_progression_total[iteration]
    for agent_id, values in current_hv_progression.items():
        if agent_id < 15:
            color = colors[agent_id]
        else:
            color = colors[agent_id + 10]
        dates = values['time']
        hv_values = values['hv']
        ax.plot(dates, hv_values, label=f'hv agent {agent_id}', color=color)
    ax.legend(loc='center right', fontsize='x-large')
    plt.xlabel('Time')
    plt.ylabel('Hypervolume')
    plt.title('Convergence Hypervolume total')
    plt.show()

    # plot hv progression after decide per agent
    fig, ax = plt.subplots()
    current_hv_progression = hv_progression_after_decide[iteration]
    for agent_id, values in current_hv_progression.items():
        if agent_id < 15:
            color = colors[agent_id]
        else:
            color = colors[agent_id + 10]
        dates = values['time']
        hv_values = values['hv']
        ax.plot(dates, hv_values, label=f'hv agent {agent_id}', color=color)
    ax.legend(loc='center right', fontsize='x-large')
    plt.xlabel('Time')
    plt.ylabel('Hypervolume')
    plt.title('Convergence Hypervolume after decide')
    plt.show()

    # plot hv progression after update per agent
    fig, ax = plt.subplots()
    current_hv_progression = hv_progression_after_update[iteration]
    for agent_id, values in current_hv_progression.items():
        if agent_id < 15:
            color = colors[agent_id]
        else:
            color = colors[agent_id + 10]
        dates = values['time']
        hv_values = values['hv']
        ax.plot(dates, hv_values, label=f'hv agent {agent_id}', color=color)
    ax.legend(loc='center right', fontsize='x-large')
    plt.xlabel('Time')
    plt.ylabel('Hypervolume')
    plt.title('Convergence Hypervolume after update')
    plt.show()

def plot_two_objective_fronts(merged_approximated_front):


    # all objectives
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # axis[0, 0].grid()
    ax.scatter(np.array(merged_approximated_front)[:, 0], np.array(merged_approximated_front)[:, 1],
               np.array(merged_approximated_front)[:, 2], s=20, c='blue')
    ax.set_title('All Objectives', loc='left')
    ax.set_xlabel('minimize deviations', labelpad=5)
    ax.set_ylabel('minimize emissions', labelpad=5)
    ax.set_zlabel('minimize uncertainties', labelpad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.show()

    figure, axis = plt.subplots(2, 2)

    selection = HyperVolumeContributionSelection(prefer_boundary_points=False)
    selection.sorting_component.hypervolume_indicator.reference_point = [1.1, 1.1]

    f1 = np.array(merged_approximated_front)[:, 0]
    f2 = np.array(merged_approximated_front)[:, 1]
    f3 = np.array(merged_approximated_front)[:, 2]

    # f1, f2
    axis[0, 1].scatter(f1, f2, s=3, c='blue')
    axis[0, 1].set_xlabel('minimize deviations', labelpad=5)
    axis[0, 1].set_ylabel('minimize emissions', labelpad=5)
    axis[0, 1].set_xlim(0, 1)
    axis[0, 1].set_ylim(0, 1)
    axis[0, 1].set_title('f1 & f2', loc='left')

    # f1, f3
    axis[1, 0].scatter(f1, f3, s=3, c='blue')
    axis[1, 0].set_xlabel('minimize deviations', labelpad=5)
    axis[1, 0].set_ylabel('minimize uncertainty', labelpad=5)
    axis[1, 0].set_xlim(0, 1)
    axis[1, 0].set_ylim(0, 1)
    axis[1, 0].set_title('f1 & f3', loc='left')

    # f2, f3
    axis[1, 1].scatter(f2, f3, s=3, c='blue')
    axis[1, 1].set_xlabel('minimize emissions', labelpad=5)
    axis[1, 1].set_ylabel('minimize uncertainty', labelpad=5)
    axis[1, 1].set_xlim(0, 1)
    axis[1, 1].set_ylim(0, 1)
    axis[1, 1].set_title('f2 & f3', loc='left')

    figure.tight_layout()

    plt.show()


def plot_ref_front(path):
    merged_approximated_front = []

    with h5py.File(path, 'r') as results:
        size = len(np.array(results['front']))-1
        for idx in range(size):
            print(idx)
            f1 = results['front']['entries' + str(idx)]['Performance_0']
            f2 = results['front']['entries' + str(idx)]['Performance_1']
            f3 = results['front']['entries' + str(idx)]['Performance_2']
            merged_approximated_front.append([f1, f2, f3])

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.grid()

    ax.scatter(np.array(merged_approximated_front)[:, 0], np.array(merged_approximated_front)[:, 1],
               np.array(merged_approximated_front)[:, 2], c='blue', s=50)
    ax.set_title('Results Ref Front')

    # Set axes label
    ax.set_xlabel('minimize deviations', labelpad=20)
    ax.set_ylabel('minimize emissions', labelpad=20)
    ax.set_zlabel('minimize uncertainties', labelpad=20)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.show()




# paths_to_use = {'results_scenario2.hdf5': 30}

# paths_to_use = {'results_scenario13_1.hdf5': 5, 'results_scenario13_2.hdf5': 5,
#                 'results_scenario13_3.hdf5': 5, 'results_scenario13_4.hdf5': 10, 'results_scenario13_5.hdf5': 5}

# paths_to_use = {'scenario2_verteilt_01.hdf5': 1, 'scenario2_verteilt_02.hdf5': 1, 'scenario2_verteilt_03.hdf5': 1,
#                 'scenario2_verteilt_04.hdf5': 1, 'scenario2_verteilt_05.hdf5': 1, 'scenario2_verteilt_06.hdf5': 1,
#                 'scenario2_verteilt_07.hdf5': 1, 'scenario2_verteilt_08.hdf5': 1, 'scenario2_verteilt_09.hdf5': 1,
#                 'scenario2_verteilt_10.hdf5': 1, 'scenario2_verteilt_11.hdf5': 1, 'scenario2_verteilt_12.hdf5': 1,
#                 'scenario2_verteilt_13.hdf5': 1, 'scenario2_verteilt_14.hdf5': 1, 'scenario2_verteilt_15.hdf5': 1,
#                 'scenario2_verteilt_16.hdf5': 1, 'scenario2_verteilt_17.hdf5': 1, 'scenario2_verteilt_18.hdf5': 1,
#                 'scenario2_verteilt_19.hdf5': 1, 'scenario2_verteilt_20.hdf5': 1, 'scenario2_verteilt_21.hdf5': 1,
#                 'scenario2_verteilt_22.hdf5': 1, 'scenario2_verteilt_23.hdf5': 1, 'scenario2_verteilt_24.hdf5': 1,
#                 'scenario2_verteilt_25.hdf5': 1, 'scenario2_verteilt_26.hdf5': 1, 'scenario2_verteilt_27.hdf5': 1,
#                 'scenario2_verteilt_28.hdf5': 1, 'scenario2_verteilt_29.hdf5': 1, 'scenario2_verteilt_30.hdf5': 1
#                 }

paths_to_use = {'scenario13_verteilt_01.hdf5': 1, 'scenario13_verteilt_02.hdf5': 1, 'scenario13_verteilt_03.hdf5': 1,
                'scenario13_verteilt_04.hdf5': 1, 'scenario13_verteilt_05.hdf5': 1, 'scenario13_verteilt_06.hdf5': 1,
                'scenario13_verteilt_07.hdf5': 1, 'scenario13_verteilt_08.hdf5': 1, 'scenario13_verteilt_09.hdf5': 1,
                'scenario13_verteilt_10.hdf5': 1, 'scenario13_verteilt_11.hdf5': 1, 'scenario13_verteilt_12.hdf5': 1
                }

# paths_to_use = {'results_scenario13_mehr_iterationen.hdf5': 1}

# paths_to_use = {'results_scenario13_fully_connected_sleep005.hdf5': 1}

# paths_to_use = {'results_scenario13_1.hdf5': 5, 'results_scenario13_2.hdf5': 5, 'results_scenario13_3.hdf5': 5,
#                 'results_scenario13_4.hdf5': 10, 'results_scenario13_5.hdf5': 5, 'results_scenario2.hdf5': 30,
#                 '2.hdf5': 1}


# create_reference_front(paths_to_use)
# analyze_results(number_of_runs=NUM_SIMULATIONS, path=DB_FILE, path_to_reference_front='ref_front.hdf5')
analyze_results(paths=paths_to_use, total_number_of_runs=12, path_to_reference_front='ref_front.hdf5')
# plot_ref_front('ref_front.hdf5')