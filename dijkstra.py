import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from past_data import get_data
from cop import COP, ceclius_to_fahrenheit

HORIZON = 48 # 11*31*24 # hours
ADD_MIN_HOURS = -5
ADD_MAX_HOURS = -5
HP_POWER = 12 # kW
M_TANKS = 454.25*3 # kg
MIN_TOP_TEMP = 50 # C
MAX_TOP_TEMP = 85 # C
TEMP_LIFT = 11 # C
TEMP_DROP = 11 # C
NUM_LAYERS = 2400
OVERESTIME_LOAD = 0 # %
PUNISH_LOW_HP = False

class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        self.energy = self.get_energy()
        self.pathcost = 0 if time_slice==HORIZON else int(1e9)
        self.next_node = None

    def __repr__(self):
        return f"Node[time_slice:{self.time_slice}, top_temp:{self.top_temp}, thermocline:{self.thermocline}]"

    def get_energy(self):
        m_layer = M_TANKS / NUM_LAYERS
        energy_top = (self.thermocline-1)*m_layer * 4187 * self.top_temp
        energy_bottom = (NUM_LAYERS-self.thermocline)*m_layer * 4187 * (self.top_temp-TEMP_LIFT)
        energy_middle = m_layer*4187*(self.top_temp-TEMP_LIFT/2)
        total_joules = energy_top + energy_bottom + energy_middle
        total_kWh = total_joules/3600/1000
        return round(total_kWh,2)


class Edge():
    def __init__(self, tail:Node, head:Node, cost:float, energy_from_HP:float):
        self.tail = tail
        self.head = head
        self.cost = cost
        self.energy_from_HP = energy_from_HP

    def __repr__(self):
        return f"Edge: {self.tail} --cost:{round(self.cost,5)}--> {self.head}"


class Graph():
    def __init__(self, start_state:Node, start_time):
        print("\nSetting up the graph...")
        timer = time.time()
        self.start_time = start_time
        self.source_node = start_state
        self.get_forecasts()
        self.define_nodes()
        self.define_edges()
        # for h in range(HORIZON+1):
        #     print(f"\nHour {h} has {len(self.nodes[h])} nodes:")
        #     for n in self.nodes[h]:
        #         print(f"-{n}")
        #     if h==1:
        #         break
        print(f"Done in {round(time.time()-timer,1)} seconds.\n")

    def get_forecasts(self):
        df = get_data(self.start_time, HORIZON)
        self.elec_prices = list(df.elec_prices)
        # self.elec_prices = list(df.jan24_prices)
        # self.elec_prices = list(df.jul24_prices)
        self.oat = list(df.oat)
        self.load = list(df.load)
        self.load = [self.load[0]] + [x*(1+OVERESTIME_LOAD/100) for x in self.load[1:]]

    def define_nodes(self):
        self.nodes = {}
        for time_slice in range(HORIZON+1):
            self.nodes[time_slice] = [self.source_node] if time_slice==0 else []
            self.nodes[time_slice].extend(
                Node(time_slice, top_temp, thermocline)
                for top_temp in range(MIN_TOP_TEMP, MAX_TOP_TEMP, TEMP_LIFT) 
                for thermocline in [1] + list(range(100, NUM_LAYERS + 1, 100))
                if (time_slice, top_temp, thermocline) != (0, self.source_node.top_temp, self.source_node.thermocline)
            )
        
        # Add the min and max nodes in the first time slice
        for h in range(HORIZON+1):
            for node in self.nodes[h]:
                # Discharging: find node that minimizes energy_from_hp
                if h<ADD_MIN_HOURS:
                    thermoc = node.thermocline
                    toptemp = node.top_temp
                    while True:
                        if thermoc==1:
                            if toptemp-TEMP_DROP < MIN_TOP_TEMP:
                                break
                            toptemp = toptemp-TEMP_DROP
                            thermoc = NUM_LAYERS
                        node_next = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                        energy_to_store = node_next.energy - node.energy
                        if energy_to_store + self.load[h] < 0:
                            thermoc += 1
                            if thermoc == NUM_LAYERS+1:
                                thermoc = 2
                            break
                        thermoc += -1
                    min_node = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                    self.nodes[h+1].append(min_node)
                # Charging: find node that maximizes energy_from_hp
                if h < ADD_MAX_HOURS:
                    thermoc = node.thermocline
                    toptemp = node.top_temp
                    while True:
                        if thermoc==NUM_LAYERS:
                            if toptemp+TEMP_LIFT > MAX_TOP_TEMP:
                                break
                            toptemp = toptemp+TEMP_LIFT
                            thermoc = 1
                        node_next = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                        energy_to_store = node_next.energy - node.energy
                        if  energy_to_store + self.load[h] > HP_POWER:
                            thermoc += -1
                            if thermoc == 0:
                                thermoc = NUM_LAYERS-1
                            break
                        thermoc += 1
                    max_node = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                    self.nodes[h+1].append(max_node)
    
    def define_edges(self):
        self.edges = {}
        for h in range(HORIZON):
            for node_now in self.nodes[h]:
                self.edges[node_now] = []
                for node_next in self.nodes[h+1]:

                    energy_to_store = node_next.energy - node_now.energy
                    energy_from_HP = energy_to_store + self.load[h]

                    losses = 0.05*(node_now.energy-Node(0,MIN_TOP_TEMP,1).energy)
                    energy_from_HP += losses

                    if energy_from_HP <= HP_POWER and energy_from_HP>-0.5:
                        
                        cop = COP(oat=self.oat[h], ewt=node_now.top_temp-TEMP_LIFT, lwt=node_next.top_temp)
                        elec_cost = self.elec_prices[h] / 1000
                        cost = elec_cost * energy_from_HP / cop

                        if PUNISH_LOW_HP:
                            if h==0 and energy_from_HP < 2 and energy_from_HP > 0.05:
                                cost += 1e9

                        if energy_to_store > 0:
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline>node_now.thermocline:
                                self.edges[node_now].append(Edge(node_now,node_next,cost,energy_from_HP))
                            if node_next.top_temp==node_now.top_temp+TEMP_LIFT:
                                self.edges[node_now].append(Edge(node_now,node_next,cost,energy_from_HP))

                        elif energy_to_store < 0:
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline<node_now.thermocline:
                                self.edges[node_now].append(Edge(node_now,node_next,cost,energy_from_HP))
                            if node_next.top_temp==node_now.top_temp-TEMP_DROP:
                                self.edges[node_now].append(Edge(node_now,node_next,cost,energy_from_HP))

                        else:
                            self.edges[node_now].append(Edge(node_now,node_next,cost,energy_from_HP))

    def solve_dijkstra(self):
        print("Solving Dijkstra...")
        start_time = time.time()
        for time_slice in range(HORIZON-1, -1, -1):
            for node in (n for n in self.nodes[time_slice] if n in self.edges):
                best_edge = min(self.edges[node], key=lambda e: e.head.pathcost + e.cost)
                if best_edge.energy_from_HP<0: 
                    best_edge_neg = max([e for e in self.edges[node] if e.energy_from_HP<0], key=lambda e: e.energy_from_HP)
                    best_edge_pos = min([e for e in self.edges[node] if e.energy_from_HP>=0], key=lambda e: e.energy_from_HP)
                    best_edge = best_edge_pos if (-best_edge_neg.energy_from_HP >= best_edge_pos.energy_from_HP) else best_edge_neg
                node.pathcost = best_edge.head.pathcost + best_edge.cost
                node.next_node = best_edge.head
        print(f"Done in {round(time.time()-start_time,3)} seconds.\n")
        return

    def plot(self, print_nodes:bool):
        self.list_hp_energy = []
        self.list_storage_energy = []
        self.list_elec_prices = []
        self.list_load = []
        self.list_thermoclines = []
        self.list_toptemps = []
        # Go through the shortest path
        node_i = self.source_node
        if print_nodes: print(node_i)
        while node_i.next_node is not None:
            energy_to_store = node_i.next_node.energy - node_i.energy
            energy_from_HP = energy_to_store + self.load[node_i.time_slice]
            losses = 0.05*(node_i.energy-Node(0,MIN_TOP_TEMP,1).energy)
            energy_from_HP += losses
            self.list_storage_energy.append(node_i.energy)
            self.list_elec_prices.append(self.elec_prices[node_i.time_slice])
            self.list_load.append(self.load[node_i.time_slice])
            self.list_hp_energy.append(round(energy_from_HP,2))
            self.list_thermoclines.append(node_i.thermocline)
            self.list_toptemps.append(node_i.top_temp)
            node_i = node_i.next_node
            if print_nodes: print(node_i)
        self.list_storage_energy.append(node_i.energy)
        self.list_thermoclines.append(node_i.thermocline)
        self.list_toptemps.append(node_i.top_temp)
        # Plot the results
        min_energy = Node(0,MIN_TOP_TEMP,1).energy
        max_energy = Node(0,MAX_TOP_TEMP,NUM_LAYERS).energy
        soc_list = [(x-min_energy)/(max_energy-min_energy)*100 for x in self.list_storage_energy]
        time_list = list(range(len(soc_list)))
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
        end_time = self.start_time.add(hours=HORIZON).format('YYYY-MM-DD HH:mm')
        fig.suptitle(f'From {self.start_time.format('YYYY-MM-DD HH:mm')} to {end_time}\nCost: {self.source_node.pathcost/10} $', fontsize=10)
        # First plot
        ax[0].step(time_list, self.list_hp_energy+[self.list_hp_energy[-1]], where='post', color='tab:blue', label='HP', alpha=0.6)
        ax[0].step(time_list, self.list_load+[self.list_load[-1]], where='post', color='tab:red', label='Load', alpha=0.6)
        ax[0].set_ylabel('Heat [kWh]')
        ax[0].set_ylim([-0.5,20])
        ax[0].legend(loc='upper left')
        ax2 = ax[0].twinx()
        ax2.step(time_list, self.list_elec_prices+[self.list_elec_prices[-1]], where='post', color='gray', alpha=0.6, label='Elec price')
        ax2.set_ylabel('Electricity price [$/MWh]')
        ax2.legend(loc='upper right')
        ax2.set_ylim([0,600])
        if len(time_list)<50 and len(time_list)>10:
            ax[1].set_xticks(list(range(0,len(time_list)+1,2)))
        # Second plot
        norm = Normalize(vmin=ceclius_to_fahrenheit(MIN_TOP_TEMP-TEMP_LIFT-10), vmax=ceclius_to_fahrenheit(MAX_TOP_TEMP))
        cmap = matplotlib.colormaps['Reds']
        inverse_list_thermoclines = [NUM_LAYERS-x+1 for x in self.list_thermoclines]
        fahrenheit_toptemps = [ceclius_to_fahrenheit(x) for x in self.list_toptemps]
        bottom_bar_colors = [cmap(norm(ceclius_to_fahrenheit(x-TEMP_LIFT))) for x in self.list_toptemps]
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
    
    # import running

    import pendulum
    time_now = pendulum.datetime(2022, 1, 1, 0, 0, 0, tz='America/New_York')
    state_now = Node(time_slice=0, top_temp=50, thermocline=600)

    g = Graph(state_now, time_now)
    g.solve_dijkstra()
    g.plot(print_nodes=False)

    # for n in g.nodes[1]:
        # n.energy
        # print(f"Path cost: {n.pathcost}, Next node: {n.next_node}")

    # for e in g.edges[g.source_node]:
    #     print(e)