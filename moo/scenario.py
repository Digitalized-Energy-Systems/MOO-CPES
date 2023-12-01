import asyncio
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from mango import RoleAgent
from mango import create_container

from mango_library.coalition.core import (
    CoalitionParticipantRole,
    CoalitionInitiatorRole,
)
from mango_library.negotiation.multiobjective_cohda.cohda_messages import (
    MoCohdaNegotiationMessage,
)
from mango_library.negotiation.multiobjective_cohda.data_classes import (
    SolutionCandidate,
    SolutionPoint,
)
from mango_library.negotiation.multiobjective_cohda.mocohda_solution_aggregation import (
    MoCohdaSolutionAggregationRole,
)
from mango_library.negotiation.multiobjective_cohda.mocohda_starting import (
    MoCohdaNegotiationDirectStarterRole,
)
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import (
    MultiObjectiveCOHDARole,
    MoCohdaNegotiationModel,
    MoCohdaNegotiation,
)
from mango_library.negotiation.termination import (
    NegotiationTerminationParticipantRole,
    NegotiationTerminationDetectorRole,
)
from moo.config import ROOT_PATH, NUM_CHPS, NUM_WIND, DB_FILE, TARGET, NUM_SIMULATIONS, WIND_CONFIG, \
    NUM_SOLUTION_POINTS
from moo.targets import TARGETS

DATA_PATH = ROOT_PATH.parent / 'data'

# parameters for wind agents
NUM_ITERATIONS_WIND = 1
# PICK_FKT_WIND = MoCohdaNegotiation.pick_random_point
PICK_FKT_WIND = MoCohdaNegotiation.pick_all_points
MUTATE_FKT_WIND = MoCohdaNegotiation.mutate_with_neighbouring_schedule

# parameters for chps
NUM_ITERATIONS_CHP = 1
PICK_FKT_CHP = MoCohdaNegotiation.pick_random_point
# PICK_FKT_CHP = MoCohdaNegotiation.pick_all_points
MUTATE_FKT_CHP = MoCohdaNegotiation.mutate_with_one_random_schedule
# MUTATE_FKT_CHP = MoCohdaNegotiation.mutate_with_all_possible

# global parameter
CHECK_MSG_QUEUE_INTERVAL = 0.05


async def energy_scenario():
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    # create hdf5 database
    with h5py.File(DB_FILE, "w") as f:
        f.close()

    # data for chps
    schedules_chp_complete = {}
    for idx in range(NUM_CHPS, NUM_CHPS + NUM_WIND):
        if idx <= NUM_CHPS + 6:
            schedules_chp = pd.read_csv(DATA_PATH / f'chp_200_kw_{idx}.csv')
            schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]
            schedules_chp_complete[idx] = schedules_chp
        else:
            schedules_chp = pd.read_csv(DATA_PATH / f'chp_400_kw_{idx}.csv')
            schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]
            schedules_chp_complete[idx] = schedules_chp

    # data for wind devices
    power_wind_complete = {}
    for idx, kw in enumerate(WIND_CONFIG):
        power_wind_complete[idx] = [pd.read_csv(DATA_PATH / f'wind_power_{kw}_kw.csv')['kw'].tolist()]

    max_capacities = {1: 200, 2: 200, 3: 250, 4: 250, 5: 250, 6: 300, 7: 300, 8: 300, 9: 350, 10: 350,
                      11: 350, 12: 350, 13: 400, 14: 400, 15: 400, 16: 200, 17: 200, 18: 200, 19: 200, 20: 200,
                      21: 200, 22: 200, 23: 400, 24: 400, 25: 400, 26: 400, 27: 400, 28: 400, 29: 400, 30: 400}

    # keys in emissions are gCO2eq/kWh for gas (combined cycle)
    # taken from https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf#page=7
    target_params = {'emissions': {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                   350: [16, 17, 18, 21, 22, 29, 30],
                                   490: [19, 20, 23, 24, 25, 26, 27, 28]},
                     'uncertainties': {'0': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                       '1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
                     'max_emissions': None,
                     'max_target_deviation': None}

    # calculate max possible value for emissions (= every unit runs with max capacity in every interval)
    max_emissions = 0
    for emission, idx_list in target_params['emissions'].items():
        for idx in idx_list:
            # emissions are in kwh, capacities in kw and we have 15min intervals, calculate for 1/4th of kwh
            max_emissions += emission * max_capacities[idx] * 0.25 * len(TARGET)

    target_params['max_emissions'] = max_emissions

    # # calculate max deviation from target based on target schedule and max capacity
    # total_max_capacity = 0
    # for unit, capacity in max_capacities.items():
    #     total_max_capacity += capacity
    # max_target_deviation = []
    # for time_interval in TARGET:
    #     # if difference between total_max_capacity and target is larger than target itself, this is max deviation
    #     if abs(time_interval - total_max_capacity) > time_interval:
    #         max_target_deviation.append(abs(time_interval - total_max_capacity))
    #     # else, max deviation is just the target itself (equal to no unit generates any power)
    #     else:
    #         max_target_deviation.append(time_interval)
    # target_params['max_target_deviation'] = max_target_deviation

    # calculate max deviation from target based on target schedule
    max_target_deviation = []

    for time_interval in TARGET:
        max_target_deviation.append(time_interval)
    target_params['max_target_deviation'] = max_target_deviation

    def schedule_provider_chp(agent_id=None):
        return schedules_chp_complete[int(agent_id) - 1]

    def schedule_provider_wind(agent_id=None):
        return power_wind_complete[int(agent_id) - 1]

    for simulation_idx in range(NUM_SIMULATIONS):
        # create container
        c = await create_container(addr=("127.0.0.3", 5555))

        # create cohda_agents
        cohda_agents = []
        addrs = []
        controller_agent = RoleAgent(c)
        termination_detector_role = NegotiationTerminationDetectorRole(
            aggregator_addr=c.addr, aggregator_id=controller_agent.aid
        )
        controller_agent.add_role(termination_detector_role)
        aggregation_role = MoCohdaSolutionAggregationRole(
            solution_point_choosing_function=choose_first_solution_point
        )
        controller_agent.add_role(aggregation_role)

        # create wind agents
        for i in range(NUM_WIND):
            a = RoleAgent(c, suggested_aid='agent_' + str(i))
            cohda_role = MultiObjectiveCOHDARole(
                schedule_provider=schedule_provider_wind,
                targets=TARGETS,
                local_acceptable_func=lambda s: True,
                num_solution_points=NUM_SOLUTION_POINTS,
                num_iterations=NUM_ITERATIONS_WIND,
                check_inbox_interval=CHECK_MSG_QUEUE_INTERVAL,
                pick_func=PICK_FKT_WIND,
                mutate_func=MUTATE_FKT_WIND,
                target_params=target_params,
                improve_inter_agent_variation=True
            )
            a.add_role(cohda_role)
            a.add_role(CoalitionParticipantRole())
            a.add_role(
                NegotiationTerminationParticipantRole(
                    negotiation_model_class=MoCohdaNegotiationModel,
                    negotiation_message_class=MoCohdaNegotiationMessage,
                )
            )
            addrs.append((c.addr, a.aid))
            cohda_agents.append(a)

        # create chp agents
        for i in range(NUM_CHPS):
            a = RoleAgent(c, suggested_aid='agent_' + str(i + 15))
            cohda_role = MultiObjectiveCOHDARole(
                schedule_provider=schedule_provider_chp,
                targets=TARGETS,
                local_acceptable_func=lambda s: True,
                num_solution_points=NUM_SOLUTION_POINTS,
                num_iterations=NUM_ITERATIONS_CHP,
                check_inbox_interval=CHECK_MSG_QUEUE_INTERVAL,
                pick_func=PICK_FKT_CHP,
                mutate_func=MUTATE_FKT_CHP,
                target_params=target_params,
                improve_inter_agent_variation=True
            )

            a.add_role(cohda_role)
            a.add_role(CoalitionParticipantRole())
            a.add_role(
                NegotiationTerminationParticipantRole(
                    negotiation_model_class=MoCohdaNegotiationModel,
                    negotiation_message_class=MoCohdaNegotiationMessage,
                )
            )
            addrs.append((c.addr, a.aid))
            cohda_agents.append(a)

        coalition_initiator_role = CoalitionInitiatorRole(
            addrs, "mocohda", "mocohda-negotiation"
        )
        controller_agent.add_role(coalition_initiator_role)

        await wait_for_assignments_sent(coalition_initiator_role)
        print("Starts negotiation", time.time())
        start_time = time.time()
        cohda_agents[0].add_role(
            MoCohdaNegotiationDirectStarterRole(
                target_params={'target_schedule': TARGET}, num_solution_points=NUM_SOLUTION_POINTS
            )
        )
        await wait_for_term(controller_agent)
        end_time = time.time()
        duration = end_time - start_time

        for a in cohda_agents + [controller_agent]:
            if a._check_inbox_task.done():
                if a._check_inbox_task.exception() is not None:
                    raise a._check_inbox_task.exception()
                else:
                    assert False, f"check_inbox terminated unexpectedly."

        await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=4000)
        final_memory = next(
            iter(
                cohda_agents[0]
                .roles[0]
                .context.get_or_create_model(MoCohdaNegotiationModel)
                ._negotiations.values()
            )
        )._memory

        sol_candidate = final_memory.solution_candidate

        f = h5py.File(DB_FILE, 'a')
        grp = f.create_group(f"iteration {str(simulation_idx)}")
        dtype_general = np.dtype(
            [
                ("Name", "S100"),
            ]
        )
        data_general = np.array(
            [
                (
                    f'sim {str(simulation_idx)}',
                )
            ],
            dtype=dtype_general,
        )
        grp.create_dataset("general_info", data=data_general)
        grp.create_dataset("targets", data=target_params['target_schedule'])
        solution = aggregation_role.complete_solution.cluster_schedules
        sol_grp = grp.create_group("solution points")
        for idx in range(len(solution)):
            sol_grp.create_dataset(f"solution_point_{idx}", data=solution[idx].tolist())
        grp.create_dataset("duration", data=duration)
        duration = time.strftime("%H:%M:%S", time.gmtime(duration))
        print(f'Negotiation done. Duration: {duration}')
        grp.create_dataset("messages", data=c.msgs)
        grp.create_dataset("hypervolume", data=sol_candidate.hypervolume)
        dtype_performances = np.dtype(
            [(f"Performance_{i}", "float64") for i, _ in enumerate(TARGETS)]
        )
        data_perf = np.array(
            sorted(sol_candidate.perf), dtype=dtype_performances
        )
        grp.create_dataset("performances", data=data_perf)
        print('container msgs ', c.msgs)
        cs_grp = grp.create_group("cluster schedules")
        for idx in range(len(sol_candidate.cluster_schedules)):
            cs_grp.create_dataset(f"cluster_schedule_{idx}", data=sol_candidate.cluster_schedules[idx].tolist())

        hv_group = grp.create_group("hypervolumes per agent total")
        for idx, agent in enumerate(cohda_agents):
            hv_dict_total = \
                list(agent.roles[0].context.get_or_create_model(MoCohdaNegotiationModel)._negotiations.values())[
                    0].hvs_total
            hv_group.create_dataset(f"hv_list_{idx}_keys", data=list(hv_dict_total.keys()))
            hv_group.create_dataset(f"hv_list_{idx}_values", data=list(hv_dict_total.values()))

        hvs_after_decide_grp = grp.create_group("hypervolumes per agent after decide")
        for idx, agent in enumerate(cohda_agents):
            hv_dict_after_decide = \
                list(agent.roles[0].context.get_or_create_model(MoCohdaNegotiationModel)._negotiations.values())[
                    0].hvs_after_decide
            hvs_after_decide_grp.create_dataset(f"hv_list_{idx}_keys", data=list(hv_dict_after_decide.keys()))
            hvs_after_decide_grp.create_dataset(f"hv_list_{idx}_values", data=list(hv_dict_after_decide.values()))

        hvs_after_update_grp = grp.create_group("hypervolumes per agent after update")
        for idx, agent in enumerate(cohda_agents):
            hv_dict_after_update = \
                list(agent.roles[0].context.get_or_create_model(MoCohdaNegotiationModel)._negotiations.values())[
                    0].hvs_after_update
            hvs_after_update_grp.create_dataset(f"hv_list_{idx}_keys", data=list(hv_dict_after_update.keys()))
            hvs_after_update_grp.create_dataset(f"hv_list_{idx}_values", data=list(hv_dict_after_update.values()))
            assert list(hv_dict_after_update.values()) == sorted(list(hv_dict_after_update.values()))
        all_agents_known_grp = grp.create_group("all agents known")
        for idx, agent in enumerate(cohda_agents):
            all_agents_known = \
                list(agent.roles[0].context.get_or_create_model(MoCohdaNegotiationModel)._negotiations.values())[
                    0].all_agents_known
            all_agents_known_grp.create_dataset(f"agents_known_agent_{idx}", data=all_agents_known)

        duration_per_iter_grp = grp.create_group("duration per iteration")
        for idx, agent in enumerate(cohda_agents):
            duration_per_iter_grp.create_dataset(f"mean_duration_{idx}",
                                                 data=np.mean(agent.roles[0]._duration_per_iteration))

        f.close()

        # shutdown
        for a in cohda_agents + [controller_agent]:
            await a.shutdown()
        await c.shutdown()


async def wait_for_solution_confirmed(aggregation_role):
    while len(aggregation_role._confirmed_cohda_solutions) == 0:
        await asyncio.sleep(0.05)
    print("Solution confirmed")


async def wait_for_assignments_sent(coalition_initiator_role):
    while not coalition_initiator_role._assignments_sent:
        await asyncio.sleep(0.05)


def choose_first_solution_point(solution_front: SolutionCandidate) -> SolutionPoint:
    """
    Chooses a SolutionPoint from the pareto front
    :param solution_front: MOCOHDA SolutionCandidate with solution front
    :return: the chosen SolutionPoint
    """
    return solution_front.solution_points[0].cluster_schedule


async def wait_for_term(controller_agent):
    # Function that will return once the first weight map of the given agent equals to one
    while (
            len(controller_agent.roles[0]._weight_map.values()) != 1
            or list(controller_agent.roles[0]._weight_map.values())[0] != 1
    ):
        await asyncio.sleep(0.05)


asyncio.run(energy_scenario())
