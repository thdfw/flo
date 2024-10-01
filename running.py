import pendulum
from dijkstra import Node
from closed_loop import closed_loop_simulation

time_now = pendulum.datetime(2022, 12, 16, 21, 0, 0, tz='America/New_York')
state_now = Node(time_slice=0, top_temp=50, thermocline=600)
simulation_hours = 48

closed_loop_simulation(time_now, state_now, simulation_hours, plot_iteration=False)