import random

import pandas as pd
from moo.config import ROOT_PATH, SCHEDULE_LENGTH, NUM_SCHEDULES

# load CHP file
DATA_PATH = ROOT_PATH.parent / 'data' / 'CHP_old'
for idx in range(15, 30):
    new_schedules = []

    if idx <= 15 + 6:
        kw = 200
        schedules_chp = pd.read_csv(DATA_PATH / f'chp_200_kw_{idx}.csv')
        schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]
    else:
        kw = 400
        schedules_chp = pd.read_csv(DATA_PATH / f'chp_400_kw_{idx}.csv')
        schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]

    # set first schedule as max schedule
    max_schedule = schedules_chp[0]

    # check if last values are very low, if yes, add number
    for index, value in enumerate(max_schedule):
        max_schedule[index] = round(value)
        if value < 50:
            max_schedule[index] = round(random.uniform(50, 150))

    new_schedules.append(max_schedule)

    # for all other schedules: make sure values are below max value
    x = 1
    while x < len(schedules_chp):
        for counter, value in enumerate(schedules_chp[x]):
            schedules_chp[x][counter] = round(value)
            while schedules_chp[x][counter] > max_schedule[counter]:
                schedules_chp[x][counter] -= 3
        new_schedules.append(schedules_chp[x])
        x += 1

    print(new_schedules)
    schedules_by_interval = []
    y = 0
    while y < len(new_schedules[0]):
        schedule_by_interval = []
        for new_schedule in new_schedules:
            schedule_by_interval.append(new_schedule[y])
        schedules_by_interval.append(schedule_by_interval)
        y += 1

    print(schedules_by_interval)

    pd.DataFrame(schedules_by_interval).to_csv(f'chp_{kw}_kw_{idx}.csv', index=False)
    print("")
