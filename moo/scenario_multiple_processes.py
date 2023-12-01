import multiprocessing
import time

import chp_agent
import wind_agent
import wind_starter

if __name__ == "__main__":
    script1 = multiprocessing.Process(target=wind_starter.main, args=(0,))
    all_agents = []
    for idx in range(1, 15):
        all_agents.append(multiprocessing.Process(target=wind_agent.main, args=(idx,)))
    for idx in range(15, 30):
        all_agents.append(multiprocessing.Process(target=chp_agent.main, args=(idx,)))

    for t in all_agents:
        t.start()

    script1.start()

    while True:
        script1.join(timeout=0)
        if script1.is_alive():
            time.sleep(1)
            continue
        else:
            for t in all_agents:
                t.join()
                t.terminate()
import multiprocessing
import time

import chp_agent
import wind_agent
import wind_starter

if __name__ == "__main__":
    script1 = multiprocessing.Process(target=wind_starter.main, args=(0,))
    all_agents = []
    for idx in range(1, 15):
        all_agents.append(multiprocessing.Process(target=wind_agent.main, args=(idx,)))
    for idx in range(15, 30):
        all_agents.append(multiprocessing.Process(target=chp_agent.main, args=(idx,)))

    for t in all_agents:
        t.start()

    script1.start()

    while True:
        script1.join(timeout=0)
        if script1.is_alive():
            time.sleep(1)
            continue
        else:
            for t in all_agents:
                t.join()
                t.terminate()
