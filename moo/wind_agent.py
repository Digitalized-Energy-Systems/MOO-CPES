import asyncio

import pandas as pd

from config import ROOT_PATH, TARGET, NUM_SOLUTION_POINTS, WIND_CONFIG, PORTS_TO_AIDS
from mango import RoleAgent
from mango import create_container
from mango.messages.codecs import JSON
from mango_library.coalition.core import (
    CoalitionParticipantRole,
)
from mango_library.negotiation.multiobjective_cohda.cohda_messages import (
    MoCohdaNegotiationMessage,
)
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import (
    MultiObjectiveCOHDARole,
    MoCohdaNegotiationModel,
    MoCohdaNegotiation,
)
from mango_library.negotiation.termination import (
    NegotiationTerminationParticipantRole,
)
from mango_library.negotiation.util import multi_objective_serializers
from targets import TARGETS

DATA_PATH = ROOT_PATH.parent / 'data'

# parameters for wind agents
NUM_ITERATIONS_WIND = 1
# PICK_FKT_WIND = MoCohdaNegotiation.pick_random_point
PICK_FKT_WIND = MoCohdaNegotiation.pick_all_points
MUTATE_FKT_WIND = MoCohdaNegotiation.mutate_with_neighbouring_schedule

# global parameter
CHECK_MSG_QUEUE_INTERVAL = 0.05


async def run_agent(pos):
    c_agent_id = PORTS_TO_AIDS[pos][1]
    addr = PORTS_TO_AIDS[pos][0]
    codec = JSON()
    for serializer in multi_objective_serializers:
        codec.add_serializer(*serializer())

    # data for wind devices
    power_wind_complete = {}
    if pos < 10:
        kw = WIND_CONFIG[int(c_agent_id[-1])]
    else:
        kw = WIND_CONFIG[int(c_agent_id[-2:])]
    power_wind_complete = [pd.read_csv(DATA_PATH / f'wind_power_{kw}_kw.csv')['kw'].tolist()]

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

    # calculate max deviation from target based on target schedule
    max_target_deviation = []

    for time_interval in TARGET:
        max_target_deviation.append(time_interval)
    target_params['max_target_deviation'] = max_target_deviation

    def schedule_provider_wind(agent_id=None):
        return power_wind_complete

    # create container
    c = await create_container(addr=addr, codec=codec)

    # create cohda_agent
    cohda_agents = []
    addrs = []

    # create wind agent
    a = RoleAgent(c, suggested_aid=c_agent_id)
    cohda_role = MultiObjectiveCOHDARole(
        schedule_provider=schedule_provider_wind,
        targets=TARGETS,
        local_acceptable_func=lambda s: True,
        num_solution_points=NUM_SOLUTION_POINTS,
        num_iterations=NUM_ITERATIONS_WIND,
        check_inbox_interval=CHECK_MSG_QUEUE_INTERVAL,
        pick_func=PICK_FKT_WIND,
        mutate_func=MUTATE_FKT_WIND,
        target_params=target_params
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

    while True:
        await asyncio.sleep(1000)
        print(a.aid, c.msgs)
    await c.shutdown()


def main(pos):
    asyncio.run(run_agent(pos))

# parser = argparse.ArgumentParser()
# args = parser.parse_known_args()
# position = args[1][-1]
# if len(position) == 6:
#     position = position[-1]
# else:
#     position = position[-2:]
# asyncio.run(run_agent(pos=int(position)))
