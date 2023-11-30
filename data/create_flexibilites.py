from os.path import abspath
from pathlib import Path

import pandas as pd
from pysimmods.other.flexibility.flexibility_model import FlexibilityModel
from pysimmods.generator.chplpgsystemsim.chplpg_system import CHPLPG
from pysimmods.generator.chplpgsystemsim.presets import chp_preset

from moo.config import START, NUM_SCHEDULES, SIMULATION_HOURS, START_FORMAT_CSV, SCHEDULE_LENGTH

# CHP schedules
kw_possibilities = [7, 14, 200, 400]
STEP_SIZE = 900


def create_schedules_chp(kw=400, chp_id=0):
    chp_params, chp_inits = chp_preset(kw)
    chp_model = CHPLPG(params=chp_params['chp'], inits=chp_inits['chp'])
    schedule_model = FlexibilityModel(chp_model, step_size=STEP_SIZE)
    schedule_model.set_now_dt(START)
    schedule_model.set_step_size(STEP_SIZE)
    chp_model.set_now_dt(START)
    chp_model.inputs.e_th_demand_set_kwh = 0
    schedule_model.inputs.e_th_demand_set_kwh = 0
    schedule_model.step()
    schedules = schedule_model.generate_schedules(START,
                                                  flexibility_horizon_hours=SIMULATION_HOURS,
                                                  num_schedules=NUM_SCHEDULES - 1)
    correct_schedules = {}
    for idx, entry in enumerate(schedules._schedules.values()):
        entry = list(entry.to_dict()['p_kw'].values())
        correct_schedules[idx] = [value * (-1) for value in entry]
        assert all((val > 0) for val in correct_schedules[idx])
    correct_schedules[NUM_SCHEDULES - 1] = [0. for _ in range(SCHEDULE_LENGTH)]
    pd.DataFrame(correct_schedules).to_csv(f'chp_{kw}_kw_{chp_id}.csv', index=False)


def create_schedules_wind(kw=600):
    # installed power in total: 20.414 MW + 1.093 MW = 21507 MW
    # Max power: 16.885 MW
    # data taken from: https://www.50hertz.com/de/Transparenz/Kennzahlen/ErzeugungEinspeisung/EinspeisungausWindenergie
    # adapt data to given MW
    ROOT_PATH = Path(abspath(__file__)).parent
    original_schedule = pd.read_csv(ROOT_PATH / 'wind.csv')
    power = original_schedule['mw']
    dates = original_schedule['date']
    new_power = []
    for power_val in power:
        # adapt to given mw value, then calculate value in kw, then
        actual_power = power_val / 21507 * (kw / 1000)
        actual_power_in_kw = actual_power * 1000
        new_power.append(actual_power_in_kw)

    idx_of_start_date = -1
    for idx, date in enumerate(dates):
        if date == START_FORMAT_CSV:
            idx_of_start_date = idx
            break
    assert idx_of_start_date != -1
    new_power = new_power[idx_of_start_date:idx_of_start_date + SCHEDULE_LENGTH]
    for entry in new_power:
        assert entry <= kw
    new_dates = dates[idx_of_start_date:idx_of_start_date + SCHEDULE_LENGTH]
    new_data = {'date': new_dates, 'kw': new_power}

    pd.DataFrame(new_data).to_csv(f'wind_power_{kw}_kw.csv', index=False)


for i in range(15, 22):
    create_schedules_chp(kw=200, chp_id=i)
for i in range(22, 30):
    create_schedules_chp(kw=400, chp_id=i)

create_schedules_wind(kw=200)
create_schedules_wind(kw=250)
create_schedules_wind(kw=300)
create_schedules_wind(kw=350)
create_schedules_wind(kw=400)
