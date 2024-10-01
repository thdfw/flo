from dijkstra import Graph, Node
from dijkstra import HORIZON, MIN_TOP_TEMP, MAX_TOP_TEMP, TEMP_LIFT, NUM_LAYERS
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cop import ceclius_to_fahrenheit
import pendulum

def closed_loop_simulation(time_now, state_now, simulation_hours, plot_iteration=False):
    list_elec_prices = []
    list_load = []
    list_hp_energy = []
    list_thermoclines = []
    list_storage_energy = []
    list_toptemps = []
    simulation_start_time = time_now
    total_cost = 0

    for i in range(simulation_hours):

        print(f"Hour {i}/{simulation_hours}")

        g = Graph(state_now, time_now)
        g.solve_dijkstra()
        if plot_iteration: 
            g.plot(print_nodes=False)

        # Move to next hour
        time_now = time_now.add(hours=1)
        state_now = g.source_node.next_node
        state_now.time_slice = 0
        # print(state_now)

        total_cost += g.source_node.pathcost - g.source_node.next_node.pathcost
        list_elec_prices.append(g.elec_prices[0])
        list_load.append(g.load[0])
        energy_to_store = g.source_node.next_node.energy() - g.source_node.energy()
        energy_from_HP = energy_to_store + g.load[0]
        list_hp_energy.append(energy_from_HP)
        list_toptemps.append(g.source_node.top_temp)
        list_thermoclines.append(g.source_node.thermocline)
        list_storage_energy.append(g.source_node.energy())
    list_storage_energy.append(g.source_node.next_node.energy())
    list_thermoclines.append(g.source_node.next_node.thermocline)
    list_toptemps.append(g.source_node.next_node.top_temp)

    # Plot
    min_energy = Node(0,MIN_TOP_TEMP,1).energy()
    max_energy = Node(0,MAX_TOP_TEMP,NUM_LAYERS).energy()
    soc_list = [(x-min_energy)/(max_energy-min_energy)*100 for x in list_storage_energy]
    time_list = list(range(len(soc_list)))
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
    end_time = simulation_start_time.add(hours=simulation_hours).format('YYYY-MM-DD HH:mm')
    total_title = f"Closed loop simulation, horizon {HORIZON} hours"
    total_title += f"\nFrom {simulation_start_time.format('YYYY-MM-DD HH:mm')} to {end_time}"
    total_title += f"\nCost: {round(total_cost/10,2)} $"
    fig.suptitle(total_title, fontsize=10)
    # First plot
    ax[0].step(time_list, list_hp_energy+[list_hp_energy[-1]], where='post', color='tab:blue', label='HP', alpha=0.6)
    ax[0].step(time_list, list_load+[list_load[-1]], where='post', color='tab:red', label='Load', alpha=0.6)
    ax[0].set_ylabel('Heat [kWh]')
    ax[0].set_ylim([-0.5,20])
    ax[0].legend(loc='upper left')
    ax2 = ax[0].twinx()
    ax2.step(time_list, list_elec_prices+[list_elec_prices[-1]], where='post', color='gray', alpha=0.6, label='Elec price')
    ax2.set_ylabel('Electricity price [$/MWh]')
    ax2.legend(loc='upper right')
    ax2.set_ylim([0,600])
    if len(time_list)<50 and len(time_list)>10:
        ax[1].set_xticks(list(range(0,len(time_list)+1,2)))
    # Second plot
    norm = Normalize(vmin=ceclius_to_fahrenheit(MIN_TOP_TEMP-TEMP_LIFT-10), vmax=ceclius_to_fahrenheit(MAX_TOP_TEMP))
    cmap = matplotlib.colormaps['Reds']
    inverse_list_thermoclines = [NUM_LAYERS-x+1 for x in list_thermoclines]
    fahrenheit_toptemps = [ceclius_to_fahrenheit(x) for x in list_toptemps]
    bottom_bar_colors = [cmap(norm(ceclius_to_fahrenheit(x-TEMP_LIFT))) for x in list_toptemps]
    ax3 = ax[1].twinx()
    ax[1].bar(time_list, inverse_list_thermoclines, color=bottom_bar_colors, alpha=0.7)
    top_part = [NUM_LAYERS-x if x<NUM_LAYERS else 0 for x in inverse_list_thermoclines]
    top_bar_colors = [cmap(norm(value)) for value in fahrenheit_toptemps]
    ax[1].bar(time_list, top_part, bottom=inverse_list_thermoclines, color=top_bar_colors, alpha=0.7)
    ax[1].set_ylim([0, NUM_LAYERS])
    ax[1].set_yticks([])
    ax[1].set_xlabel('Time [hours]')
    ax[1].set_ylabel('Storage state')
    ax3.plot(time_list, soc_list, color='black', alpha=0.4, label='SoC')
    ax3.set_ylim([-1,101])
    ax3.set_ylabel('SOC [%]')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.025, pad=0.15, alpha=0.7)
    cbar.set_label('Temperature [F]')
    plt.show()


if __name__ == '__main__':

    time_now = pendulum.datetime(2022, 12, 16, 21, 0, 0, tz='America/New_York')
    state_now = Node(time_slice=0, top_temp=50, thermocline=600)
    simulation_hours = 24

    closed_loop_simulation(time_now, state_now, simulation_hours, plot_iteration=False)