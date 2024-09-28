import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from past_data import get_data
from cop import COP, ceclius_to_fahrenheit
import pendulum

HORIZON = 2*24 # hours
ADD_MIN_HOURS = 8
ADD_MAX_HOURS = 1

HP_POWER = 12 # kW
M_TANKS = 454.25*3 # kg
MIN_TOP_TEMP = 50 # C
MAX_TOP_TEMP = 85 # C
TEMP_LIFT = 11 # C
TEMP_DROP = 11 # C
NUM_LAYERS = 1200
ADD_MIN_MAX = True

class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        self.pathcost = int(1e9)
        self.next_node = None

    def __repr__(self):
        return f"Node[time_slice:{self.time_slice}, top_temp:{self.top_temp}, thermocline:{self.thermocline}, pathcost:{self.pathcost}]"

    def energy(self):
        m_layer = M_TANKS / NUM_LAYERS
        energy_top = (self.thermocline-1)*m_layer * 4187 * self.top_temp
        energy_bottom = (NUM_LAYERS-self.thermocline)*m_layer * 4187 * (self.top_temp-TEMP_LIFT)
        energy_middle = m_layer*4187*(self.top_temp-TEMP_LIFT/2)
        total_joules = energy_top + energy_bottom + energy_middle
        total_kWh = total_joules/3600/1000
        return round(total_kWh,2)


class Edge():
    def __init__(self, tail:Node, head:Node, cost:float):
        self.tail = tail
        self.head = head
        self.cost = cost

    def __repr__(self):
        return f"Edge: {self.tail} --cost:{round(self.cost,1)}--> {self.head}"


class Graph():
    def __init__(self, start_state:Node, start_time):
        print("\nSetting up the graph...")
        timer = time.time()
        self.start_time = start_time
        self.source_node = start_state
        self.get_forecasts()
        self.define_nodes()
        self.define_edges()
        print(f"Done in {round(time.time()-timer)} seconds.\n")

    def get_forecasts(self):
        df = get_data(self.start_time, HORIZON)
        self.elec_prices = list(df.elec_prices)
        #self.elec_prices = list(df.jan24_prices)
        #self.elec_prices = list(df.jul24_prices)
        self.load = list(df.load)
        self.oat = list(df.oat)
        self.load = [x*0.7 for x in self.load]

    def define_nodes(self):
        self.nodes = [self.source_node]

        # Find all the allowed top temperatures
        allowed_top_slice = [self.source_node.top_temp]
        t = self.source_node.top_temp
        while t+TEMP_LIFT <= MAX_TOP_TEMP:
            t = t + TEMP_LIFT
            allowed_top_slice.append(t)
        t = self.source_node.top_temp
        while t-TEMP_LIFT >= MIN_TOP_TEMP:
            t = t - TEMP_LIFT
            allowed_top_slice.append(t)
        allowed_top_slice = sorted(allowed_top_slice)

        self.nodes.extend([
            Node(time_slice, top_temp, thermocline) 
            for time_slice in range(1,HORIZON+1) 
            for top_temp in allowed_top_slice 
            for thermocline in range(1, NUM_LAYERS+1, 100)
            if (time_slice, 
                top_temp, 
                thermocline) != (self.source_node.time_slice, 
                                 self.source_node.top_temp, 
                                 self.source_node.thermocline)
            ])
        
        if ADD_MIN_MAX:
            # Add the min and max nodes for each node in the first time steps
            for h in range(HORIZON+1):
                # print(f"Hour {h}...")
                for node in [x for x in self.nodes if x.time_slice==h]:
                    
                    if h<=ADD_MIN_HOURS:
                        # Discharging: find node that minimizes energy_from_hp
                        thermoc = node.thermocline
                        toptemp = node.top_temp
                        while True:
                            # If the thermocline has reached the top, time to change top temp and thermocline
                            if thermoc==1:
                                toptemp = toptemp-TEMP_DROP
                                thermoc = NUM_LAYERS
                            # Set up a new node with less energy
                            node_next = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                            energy_to_store = node_next.energy() - node.energy()
                            # Stop if the storage is discharged by more than the amount of the load
                            if energy_to_store + self.load[h] < 0:
                                    thermoc += 1
                                    break
                            thermoc += -1
                        min_node = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                        self.nodes.append(min_node)

                    if h <= ADD_MAX_HOURS:
                        # Charging: find node that minimizes HP_POWER - energy_from_hp
                        thermoc = node.thermocline
                        toptemp = node.top_temp
                        while True:
                            # If the thermocline has reached the bottom, time to change top temp and thermocline
                            if thermoc==NUM_LAYERS:
                                toptemp = toptemp+TEMP_LIFT
                                thermoc = 1
                            # Set up a new node with more energy
                            node_next = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                            energy_to_store = node_next.energy() - node.energy()
                            # Stop if the storage is charged by more than the HP can provide
                            if HP_POWER - (energy_to_store + self.load[h]) < 0:
                                    thermoc += -1
                                    break
                            thermoc += 1
                        max_node = Node(h+1, thermocline=thermoc, top_temp=toptemp)
                        self.nodes.append(max_node)
    
    def define_edges(self):
        self.edges = []
        for time in range(HORIZON):
            for node_now in [x for x in self.nodes if x.time_slice==time]:
                for node_next in [x for x in self.nodes if x.time_slice==time+1]:

                    energy_to_store = node_next.energy() - node_now.energy() # int
                    energy_from_HP = energy_to_store + self.load[node_now.time_slice]

                    if energy_from_HP <= HP_POWER and energy_from_HP >= 0:
                        
                        # Compute the cost
                        cop = COP(oat=self.oat[node_now.time_slice], ewt=node_now.top_temp-TEMP_LIFT, lwt=node_next.top_temp)
                        elec_cost = self.elec_prices[node_now.time_slice] / 1000
                        cost = round(elec_cost * energy_from_HP / cop,2)

                        # CHARGING the storage
                        if energy_to_store > 0:
                            # Option 1
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline>node_now.thermocline:
                                self.edges.append(Edge(node_next,node_now,cost))
                            # Option 2
                            if node_next.top_temp==node_now.top_temp+TEMP_LIFT:
                                self.edges.append(Edge(node_next,node_now,cost))

                        # DISCHARGING the storage
                        elif energy_to_store < 0:
                            # Option 1
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline<node_now.thermocline:
                                self.edges.append(Edge(node_next,node_now,cost))
                            # Option 2
                            if node_next.top_temp==node_now.top_temp-TEMP_DROP:
                                self.edges.append(Edge(node_next,node_now,cost))

                        # DONT TOUCH the storage
                        elif energy_to_store==0 and node_next.top_temp==node_now.top_temp and node_next.thermocline==node_now.thermocline:
                            self.edges.append(Edge(node_next,node_now,cost))
   
    def solve_dijkstra(self):
        print("Solving Dijkstra...")
        start_time = time.time()

        # Initialise the nodes in the last time slice with a path cost of 0
        for node in [x for x in self.nodes if x.time_slice==HORIZON]:
            node.pathcost = 0

        # Moving backwards from the end of the horizon to current time 0
        for h in range(1, HORIZON+1):
            time_slice = HORIZON - h
            # print(f"- Working on hour {time_slice}...")

            # For all nodes in the current time slice
            for node in [x for x in self.nodes if x.time_slice==time_slice]:

                # For all edges arriving at this node
                available_edges = [e for e in self.edges if e.head==node]
                total_costs = [e.tail.pathcost+e.cost for e in available_edges]
                if available_edges == []: continue                  

                # Find which of the available edges has the minimal total cost
                best_edge = available_edges[total_costs.index(min(total_costs))]

                # Update the current node accordingly
                node.pathcost = round(min(total_costs),2)
                node.next_node = best_edge.tail

        print(f"Done in {round(time.time()-start_time)} seconds.\n")
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
            energy_to_store = node_i.next_node.energy() - node_i.energy() # int
            energy_from_HP = energy_to_store + self.load[node_i.time_slice]
            self.list_storage_energy.append(node_i.energy())
            self.list_elec_prices.append(self.elec_prices[node_i.time_slice])
            self.list_load.append(self.load[node_i.time_slice])
            self.list_hp_energy.append(round(energy_from_HP,2))
            self.list_thermoclines.append(node_i.thermocline)
            self.list_toptemps.append(node_i.top_temp)
            node_i = node_i.next_node
            if print_nodes: print(node_i)
        self.list_storage_energy.append(node_i.energy())
        self.list_thermoclines.append(node_i.thermocline)
        self.list_toptemps.append(node_i.top_temp)
        # Plot the results
        min_energy = Node(0,MIN_TOP_TEMP,1).energy()
        max_energy = Node(0,MAX_TOP_TEMP,NUM_LAYERS).energy()
        soc_list = [(x-min_energy)/(max_energy-min_energy)*100 for x in self.list_storage_energy]
        time_list = list(range(len(soc_list)))
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
        end_time = self.start_time.add(hours=HORIZON).format('YYYY-MM-DD HH:mm')
        fig.suptitle(f'From {self.start_time.format('YYYY-MM-DD HH:mm')} to {end_time}\nCost: {self.source_node.pathcost} $', fontsize=10)
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
        norm = Normalize(vmin=ceclius_to_fahrenheit(MIN_TOP_TEMP-TEMP_LIFT), vmax=ceclius_to_fahrenheit(MAX_TOP_TEMP))
        cmap = matplotlib.colormaps['Reds']
        inverse_list_thermoclines = [NUM_LAYERS-x+1 for x in self.list_thermoclines]
        fahrenheit_toptemps = [ceclius_to_fahrenheit(x) for x in self.list_toptemps]
        bottom_bar_colors = [cmap(norm(value-TEMP_LIFT)) for value in fahrenheit_toptemps]
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


time_now = pendulum.datetime(2022, 12, 16, 16, 0, 0, tz='America/New_York')
state_now = Node(time_slice=0, top_temp=50, thermocline=600)

for i in range(24):

    g = Graph(state_now, time_now)
    g.solve_dijkstra()
    g.plot(print_nodes=False)

    time_now = time_now.add(hours=1)
    state_now = g.source_node.next_node
    state_now.time_slice = 0
