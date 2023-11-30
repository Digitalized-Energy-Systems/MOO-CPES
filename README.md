This repository contains an example for a multi-objective optimization in Cyber-Physical Energy Systems. In this case,
agents represent chp units or wind plants. Regarding their flexibility, the data and scripts in the "data" folder can
be used, containing example schedules for chp units with different sizes and example wind power for wind power plants
in different sizes. To create other flexibilities, "create_flexibilties.py" can be used.
For the optimization using the agents, the agents have to fulfill a target schedule, with contains of the sum of all
unit schedules. Regarding the target schedule, three objectives are considered: minimizing the difference between the
produced power in sum and the given target schedule, minimizing the emissions and minimizing the uncertainties
(see src/targets.py).
To run the scenario, src/scenario.py can be called. The configuration regarding the settings of the units (number of
wind agents, number of chp agents, sizes of the units) and the scenario (start date, etc.) can be set in src/config.py.
To run the scenario with each agent living distributed in its own process, src/scenario_multiple_processes.py can be
used. The scenario produces a hdf5-file, containing the simulation results.