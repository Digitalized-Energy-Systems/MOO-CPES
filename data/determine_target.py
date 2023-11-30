import pandas as pd
from numpy import mean
import numpy as np

from moo.config import ROOT_PATH, SCHEDULE_LENGTH, NUM_SCHEDULES

DATA_PATH = ROOT_PATH.parent / 'data'

# power_sum_per_time = [0 for _ in range(SCHEDULE_LENGTH)]
#
# for idx in range(15, 30):
#     if idx <= 15 + 6:
#         schedules_chp = pd.read_csv(DATA_PATH / f'chp_200_kw_{idx}.csv')
#         schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]
#         mean_schedule = []
#         for s_id in range(SCHEDULE_LENGTH):
#             val = [schedules_chp[new][s_id] for new in range(NUM_SCHEDULES)]
#             mean_schedule.append(mean(val))
#         power_sum_per_time = [x + y for x, y in zip(mean_schedule, power_sum_per_time)]
#     else:
#         schedules_chp = pd.read_csv(DATA_PATH / f'chp_400_kw_{idx}.csv')
#         schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]
#         mean_schedule = []
#         for s_id in range(SCHEDULE_LENGTH):
#             val = [schedules_chp[new][s_id] for new in range(NUM_SCHEDULES)]
#             mean_schedule.append(mean(val))
#         power_sum_per_time = [x + y for x, y in zip(mean_schedule, power_sum_per_time)]
#
# print('Target schedule')
# print(power_sum_per_time)

# wind_0 = pd.read_csv(DATA_PATH / 'wind_power_200_kw.csv')['kw'].tolist()
# wind_1 = pd.read_csv(DATA_PATH / 'wind_power_200_kw.csv')['kw'].tolist()
# wind_2 = pd.read_csv(DATA_PATH / 'wind_power_250_kw.csv')['kw'].tolist()
# wind_3 = pd.read_csv(DATA_PATH / 'wind_power_250_kw.csv')['kw'].tolist()
# wind_4 = pd.read_csv(DATA_PATH / 'wind_power_250_kw.csv')['kw'].tolist()
# wind_5 = pd.read_csv(DATA_PATH / 'wind_power_300_kw.csv')['kw'].tolist()
# wind_6 = pd.read_csv(DATA_PATH / 'wind_power_300_kw.csv')['kw'].tolist()
# wind_7 = pd.read_csv(DATA_PATH / 'wind_power_300_kw.csv')['kw'].tolist()
# wind_8 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
# wind_9 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
# wind_10 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
# wind_11 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
# wind_12 = pd.read_csv(DATA_PATH / 'wind_power_400_kw.csv')['kw'].tolist()
# wind_13 = pd.read_csv(DATA_PATH / 'wind_power_400_kw.csv')['kw'].tolist()
# wind_14 = pd.read_csv(DATA_PATH / 'wind_power_400_kw.csv')['kw'].tolist()
#
#
# for i in range(SCHEDULE_LENGTH):
#     print('***')
#     print(power_sum_per_time[i])
#     sum_wind = wind_0[i] + wind_2[i] + wind_1[i] + wind_3[i] + wind_4[i] + wind_5[i] + wind_6[i] + wind_7[i] + wind_8[
#         i] + wind_9[i] + wind_10[i] + wind_11[i] + wind_12[i] + wind_13[i] + wind_14[i]
#     print(sum_wind)
#     total = power_sum_per_time[i] - sum_wind
#     assert sum_wind >= power_sum_per_time[i]
#     assert total <= 0
#     print(total)

# choose first of each CHP (max schedule) -> sum is target schedule
schedule_id = 0
target_schedule = [0 for _ in range(SCHEDULE_LENGTH)]

for idx in range(15, 30):
    if idx <= 15 + 6:
        schedules_chp = pd.read_csv(DATA_PATH / f'chp_200_kw_{idx}.csv')
        schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]
        chosen_schedule = schedules_chp[schedule_id]
        target_schedule = [x + y for x, y in zip(chosen_schedule, target_schedule)]
    else:
        schedules_chp = pd.read_csv(DATA_PATH / f'chp_400_kw_{idx}.csv')
        schedules_chp = [schedules_chp[str(i)].tolist() for i in range(10)]
        chosen_schedule = schedules_chp[schedule_id]
        target_schedule = [x + y for x, y in zip(chosen_schedule, target_schedule)]

print('Target schedule')
print(target_schedule)

# sum up all max values of wind for each interval
wind_0 = pd.read_csv(DATA_PATH / 'wind_power_200_kw.csv')['kw'].tolist()
wind_1 = pd.read_csv(DATA_PATH / 'wind_power_200_kw.csv')['kw'].tolist()
wind_2 = pd.read_csv(DATA_PATH / 'wind_power_250_kw.csv')['kw'].tolist()
wind_3 = pd.read_csv(DATA_PATH / 'wind_power_250_kw.csv')['kw'].tolist()
wind_4 = pd.read_csv(DATA_PATH / 'wind_power_250_kw.csv')['kw'].tolist()
wind_5 = pd.read_csv(DATA_PATH / 'wind_power_300_kw.csv')['kw'].tolist()
wind_6 = pd.read_csv(DATA_PATH / 'wind_power_300_kw.csv')['kw'].tolist()
wind_7 = pd.read_csv(DATA_PATH / 'wind_power_300_kw.csv')['kw'].tolist()
wind_8 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
wind_9 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
wind_10 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
wind_11 = pd.read_csv(DATA_PATH / 'wind_power_350_kw.csv')['kw'].tolist()
wind_12 = pd.read_csv(DATA_PATH / 'wind_power_400_kw.csv')['kw'].tolist()
wind_13 = pd.read_csv(DATA_PATH / 'wind_power_400_kw.csv')['kw'].tolist()
wind_14 = pd.read_csv(DATA_PATH / 'wind_power_400_kw.csv')['kw'].tolist()

wind_schedule = []
for i in range(SCHEDULE_LENGTH):
    sum_wind = wind_0[i] + wind_2[i] + wind_1[i] + wind_3[i] + wind_4[i] + wind_5[i] + wind_6[i] + wind_7[i] + wind_8[
        i] + wind_9[i] + wind_10[i] + wind_11[i] + wind_12[i] + wind_13[i] + wind_14[i]
    wind_schedule.append(sum_wind)

print("Max Wind Schedule")
print(wind_schedule)

print("difference")
print(np.subtract(target_schedule, wind_schedule))


# check difference between max values and target