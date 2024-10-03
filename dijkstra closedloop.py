import time
import pendulum
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from past_data import get_data
from cop import COP, to_celcius

HORIZON = 48 # hours
NUM_LAYERS = 2400
MIN_TOP_TEMP = 120 # F
MAX_TOP_TEMP = 180 # F
TEMP_LIFT = 20 # F
TEMP_DROP = 20 # F
HP_POWER = 12 # kW
MASS_TANKS = 454.25*3 # kg
SPECIFIC_HEAT_WATER = 4.187/3600 # kWh/kg/K

OVERESTIME_LOAD = 0 # %
PUNISH_LOW_HP = False
ADD_MIN_HOURS = -5
ADD_MAX_HOURS = -5


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
        m_layer = MASS_TANKS / NUM_LAYERS
        energy_above = self.thermocline*m_layer * SPECIFIC_HEAT_WATER * to_celcius(self.top_temp)
        energy_below = (NUM_LAYERS-self.thermocline)*m_layer * SPECIFIC_HEAT_WATER * to_celcius(self.top_temp - TEMP_LIFT)
        return energy_above + energy_below


class Edge():
    def __init__(self, tail:Node, head:Node, cost:float, heat_output_HP:float):
        self.tail = tail
        self.head = head
        self.cost = cost
        self.heat_output_HP = heat_output_HP

    def __repr__(self):
        return f"Edge: {self.tail} --cost:{round(self.cost,3)}--> {self.head}"


class Graph():
    def __init__(self, start_state:Node, start_time:pendulum.DateTime):
        print("\nSetting up the graph...")
        timer = time.time()
        self.start_time = start_time
        self.source_node = start_state
        self.get_forecasts()
        self.create_nodes()
        self.create_edges()
        # for h in range(HORIZON+1):
        #     print(f"Hour {h} has {len(self.nodes[h])} nodes")
        #     for n in self.nodes[h]:
        #         print(f"-{n}")
        #     print('')
        #     if h==1:
        #         break
        print(f"Done in {round(time.time()-timer,1)} seconds.\n")

    def get_forecasts(self):
        df = get_data(self.start_time, HORIZON)
        self.elec_prices = list(df.elec_prices)
        self.elec_prices = list(df.jan24_prices)
        # self.elec_prices = list(df.jul24_prices)
        self.oat = list(df.oat)
        self.load = list(df.load)
        self.load = [self.load[0]] + [x*(1+OVERESTIME_LOAD/100) for x in self.load[1:]]

    def create_nodes(self):
        self.nodes = {}
        for time_slice in range(HORIZON+1):
            self.nodes[time_slice] = [self.source_node] if time_slice==0 else []
            self.nodes[time_slice].extend(
                Node(time_slice, top_temp, thermocline)
                for top_temp in range(MIN_TOP_TEMP, MAX_TOP_TEMP+TEMP_LIFT, TEMP_LIFT) 
                for thermocline in [1] + list(range(100, NUM_LAYERS + 1, 100))
                if (time_slice, top_temp, thermocline) != (0, self.source_node.top_temp, self.source_node.thermocline)
            )
        
        # Add the min and max nodes in the first time slice
        for h in range(HORIZON+1):
            for node in self.nodes[h]:
                # Discharging: find node that minimizes heat_output_HP
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
                # Charging: find node that maximizes heat_output_HP
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
    
    def create_edges(self):
        self.edges = {}
        min_energy = Node(0,MIN_TOP_TEMP,1).energy
        for h in range(HORIZON):
            for node_now in self.nodes[h]:
                self.edges[node_now] = []
                for node_next in self.nodes[h+1]:
                    heat_to_store = node_next.energy - node_now.energy
                    losses = 0.005*(node_now.energy-min_energy)
                    heat_output_HP = heat_to_store + self.load[h] + losses
                    if heat_output_HP <= HP_POWER and heat_output_HP>-0.5:
                        cop = COP(oat=self.oat[h], lwt=to_celcius(node_next.top_temp))
                        cost = self.elec_prices[h]/100 * heat_output_HP / cop
                        if PUNISH_LOW_HP and h==0 and heat_output_HP < 2 and heat_output_HP > 0.05:
                                cost += 1e9
                        if heat_to_store > 0:
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline>node_now.thermocline:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                            if node_next.top_temp==node_now.top_temp+TEMP_LIFT:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                        elif heat_to_store < 0:
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline<node_now.thermocline:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                            if node_next.top_temp==node_now.top_temp-TEMP_DROP:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                        else:
                            self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))

    def solve_dijkstra(self):
        print("Solving Dijkstra...")
        start_time = time.time()
        for time_slice in range(HORIZON-1, -1, -1):
            for node in self.nodes[time_slice]:
                best_edge = min(self.edges[node], key=lambda e: e.head.pathcost + e.cost)
                if best_edge.heat_output_HP<0: 
                    best_edge_neg = max([e for e in self.edges[node] if e.heat_output_HP<0], key=lambda e: e.heat_output_HP)
                    best_edge_pos = min([e for e in self.edges[node] if e.heat_output_HP>=0], key=lambda e: e.heat_output_HP)
                    best_edge = best_edge_pos if (-best_edge_neg.heat_output_HP >= best_edge_pos.heat_output_HP) else best_edge_neg
                node.pathcost = best_edge.head.pathcost + best_edge.cost
                node.next_node = best_edge.head
        print(f"Done in {round(time.time()-start_time,3)} seconds.\n")

    def compute_bid(self):
        max_edge = max(self.edges[self.source_node], key=lambda x: x.cost)
        min_edge = min(self.edges[self.source_node], key=lambda x: x.cost)
        cop_max_edge = COP(oat=self.oat[0], lwt=to_celcius(max_edge.head.top_temp))
        cop_min_edge = COP(oat=self.oat[0], lwt=to_celcius(min_edge.head.top_temp))
        max_edge_elec_input_HP = max_edge.heat_output_HP / cop_max_edge
        min_edge_elec_input_HP = min_edge.heat_output_HP / cop_min_edge
        bid = (min_edge.head.pathcost - max_edge.head.pathcost) / (max_edge_elec_input_HP - min_edge_elec_input_HP)
        if min_edge.head.top_temp==MIN_TOP_TEMP and min_edge.head.thermocline==1 and min_edge.heat_output_HP>1:
            print(f"Warning: The house will go cold if we don't buy now (will not receive missing {round(min_edge.heat_output_HP,1)} kWh)")
        self.bid = bid
        print(f"Buy electricity if it costs less than {round(bid*100,2)} cts/kWh\n")

    def plot(self):
        self.list_toptemps = []
        self.list_thermoclines = []
        self.list_hp_energy = []
        self.list_storage_energy = []
        node_i = self.source_node
        while node_i.next_node is not None:
            edge_i = [e for e in self.edges[node_i] if e.head==node_i.next_node][0]
            self.list_toptemps.append(node_i.top_temp)
            self.list_thermoclines.append(node_i.thermocline)
            self.list_hp_energy.append(edge_i.heat_output_HP)
            self.list_storage_energy.append(node_i.energy)
            node_i = node_i.next_node
        self.list_toptemps.append(node_i.top_temp)
        self.list_thermoclines.append(node_i.thermocline)
        self.list_hp_energy.append(edge_i.heat_output_HP)
        self.list_storage_energy.append(node_i.energy)
        min_energy = Node(0,MIN_TOP_TEMP,1).energy
        max_energy = Node(0,MAX_TOP_TEMP,NUM_LAYERS).energy
        list_soc = [(x-min_energy)/(max_energy-min_energy)*100 for x in self.list_storage_energy]
        # Plot the shortest path
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
        begin = self.start_time.format('YYYY-MM-DD HH:mm')
        end = self.start_time.add(hours=HORIZON).format('YYYY-MM-DD HH:mm')
        fig.suptitle(f'From {begin} to {end}\nCost: {round(self.source_node.pathcost,2)} $', fontsize=10)
        list_time = list(range(len(list_soc)))
        # Top plot
        ax[0].step(list_time, self.list_hp_energy, where='post', color='tab:blue', label='HP', alpha=0.6)
        ax[0].step(list_time, self.load, where='post', color='tab:red', label='Load', alpha=0.6)
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Heat [kWh]')
        ax[0].set_ylim([-0.5,20])
        ax2 = ax[0].twinx()
        ax2.step(list_time, self.elec_prices, where='post', color='gray', alpha=0.6, label='Elec price')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Electricity price [cts/kWh]')
        if min(self.elec_prices)>0: ax2.set_ylim([0,60])
        # Bottom plot
        norm = Normalize(vmin=MIN_TOP_TEMP-TEMP_LIFT, vmax=MAX_TOP_TEMP)
        cmap = matplotlib.colormaps['Reds']
        tank_top_colors = [cmap(norm(x)) for x in self.list_toptemps]
        tank_bottom_colors = [cmap(norm(x-TEMP_LIFT)) for x in self.list_toptemps]
        list_thermoclines_reversed = [NUM_LAYERS-x+1 for x in self.list_thermoclines]
        ax[1].bar(list_time, list_thermoclines_reversed, color=tank_bottom_colors, alpha=0.7)
        ax[1].bar(list_time, self.list_thermoclines, bottom=list_thermoclines_reversed, color=tank_top_colors, alpha=0.7)
        ax[1].set_xlabel('Time [hours]')
        ax[1].set_ylabel('Storage state')
        ax[1].set_ylim([0, NUM_LAYERS])
        ax[1].set_yticks([])
        if len(list_time)>10 and len(list_time)<50:
            ax[1].set_xticks(list(range(0,len(list_time)+1,2)))
        ax3 = ax[1].twinx()
        ax3.plot(list_time, list_soc, color='black', alpha=0.4, label='SoC')
        ax3.set_ylabel('State of charge [%]')
        ax3.set_ylim([-1,101])
        # Color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.025, pad=0.15, alpha=0.7)
        cbar.set_label('Temperature [F]')
        plt.show()


if __name__ == '__main__':
    
    time_now = pendulum.datetime(2022, 1, 1, 0, 0, 0, tz='America/New_York')
    state_now = Node(time_slice=0, top_temp=120, thermocline=600)

    g = Graph(state_now, time_now)
    g.solve_dijkstra()
    g.compute_bid()
    # g.plot()