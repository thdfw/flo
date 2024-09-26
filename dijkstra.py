import time
import matplotlib.pyplot as plt

HORIZON = 24 # hours
HP_POWER = 12 # kW
M_LAYER = 113 # kg
MIN_TOP_TEMP = 50 # C
MAX_TOP_TEMP = 85 # C
TEMP_LIFT = 11 # C
NUM_LAYERS = 12
PRINT = False


class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        # Initial values for Dijkstra
        self.pathcost = int(1e5)
        self.next_node = None

    def __repr__(self):
        return f"Node[time_slice:{self.time_slice}, top_temp:{self.top_temp}, thermocline:{self.thermocline}]"

    def energy(self):
        energy_top = (self.thermocline-1)*M_LAYER * 4187 * self.top_temp
        energy_bottom = (NUM_LAYERS-self.thermocline)*M_LAYER * 4187 * (self.top_temp-TEMP_LIFT)
        energy_middle = M_LAYER*4187*(self.top_temp-TEMP_LIFT/2)
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
    def __init__(self, current_state):
        print("\nSetting up the graph...")
        start_time = time.time()
        self.get_forecasts()
        self.define_nodes(current_state)
        self.define_edges()
        self.list_hp_energy = []
        self.list_storage_energy = []
        self.list_elec_prices = []
        self.list_load = []
        self.list_thermoclines = []
        print(f"Done in {round(time.time()-start_time)} seconds.\n")

    def get_forecasts(self):
        self.elec_prices = [7.92, 6.63, 6.31, 6.79, 8.01, 11.58, 19.38, 21.59, 11.08, 4.49, 1.52, 
                           0.74, 0.42, 0.71, 0.97, 2.45, 3.79, 9.56, 20.51, 28.26, 23.49, 18.42, 13.23, 10.17]*3
        self.oat = [-2]*HORIZON*2
        self.load = [4]*HORIZON*2

    def COP(self, oat, ewt, lwt):
        return 2

    def define_nodes(self, current_state):
        time_slice0, top_temp0, thermocline0 = current_state
        self.source_node = Node(time_slice0, top_temp0, thermocline0)
        self.nodes = [self.source_node]
        self.nodes.extend([
            Node(time_slice, top_temp, thermocline) 
            for time_slice in range(HORIZON+1) 
            for top_temp in range(MIN_TOP_TEMP, MAX_TOP_TEMP+1) 
            for thermocline in range(1, NUM_LAYERS+1)
            if (time_slice, top_temp, thermocline) != (time_slice0, top_temp0, thermocline0)
            ])
    
    def define_edges(self):
        self.edges = []
        for time in range(HORIZON):
            for node1 in [x for x in self.nodes if x.time_slice==time]:
                for node2 in [x for x in self.nodes if x.time_slice==time+1]:

                    energy_to_store = node2.energy() - node1.energy()
                    energy_from_HP = energy_to_store + self.load[node1.time_slice]

                    if energy_from_HP <= HP_POWER and energy_from_HP >= 0:
                        
                        # Compute the cost
                        cop = self.COP(oat=self.oat[node1.time_slice], ewt=node1.top_temp-TEMP_LIFT, lwt=node2.top_temp)
                        elec_cost = self.elec_prices[node1.time_slice]
                        cost = round(elec_cost * energy_from_HP / cop,2)

                        # CHARGING the storage
                        if energy_to_store > 0:
                            # Option 1
                            if node2.top_temp==node1.top_temp and node2.thermocline>node1.thermocline:
                                self.edges.append(Edge(node2,node1,cost))
                            # Option 2
                            if node2.top_temp==node1.top_temp+TEMP_LIFT:
                                self.edges.append(Edge(node2,node1,cost))

                        # DISCHARGING the storage
                        elif energy_to_store < 0:
                            # Option 1
                            if node2.top_temp==node1.top_temp and node2.thermocline<node1.thermocline:
                                self.edges.append(Edge(node2,node1,cost))
                            # Option 2
                            if node2.top_temp==node1.top_temp-TEMP_LIFT:
                                self.edges.append(Edge(node2,node1,cost))

                        # DONT TOUCH the storage
                        elif energy_to_store == 0 and node2.top_temp==node1.top_temp and node2.thermocline==node1.thermocline:
                            self.edges.append(Edge(node2,node1,cost))
   
    def solve_dijkstra(self):
        print("Solving Dijkstra...")
        start_time = time.time()

        # Initialise the nodes in the last time slice with a path cost of 0
        for node in [x for x in self.nodes if x.time_slice==HORIZON]:
            node.pathcost = 0

        # Moving backwards from the end of the horizon to current time 0
        for h in range(1, HORIZON+1):
            time_slice = HORIZON - h
            print(f"- Working on hour {time_slice}...")

            # For all nodes in the current time slice
            for node in [x for x in self.nodes if x.time_slice==time_slice]:

                # For all edges arriving at this node
                available_edges = [e for e in self.edges if e.head==node]
                total_costs = [e.tail.pathcost+e.cost for e in available_edges]                    

                # Find which of the available edges has the minimal total cost
                best_edge = available_edges[total_costs.index(min(total_costs))]

                # Update the current node with the right dist
                node.pathcost = round(min(total_costs),2)
                node.next_node = best_edge.tail

        # Go through the shortest path
        node_i = self.source_node
        while node_i.next_node is not None:
            energy_to_store = node_i.next_node.energy() - node_i.energy()
            energy_from_HP = energy_to_store + self.load[node_i.time_slice]
            self.list_storage_energy.append(node_i.energy())
            self.list_elec_prices.append(self.elec_prices[node_i.time_slice])
            self.list_load.append(self.load[node_i.time_slice])
            self.list_hp_energy.append(round(energy_from_HP,2))
            self.list_thermoclines.append(node_i.thermocline)
            node_i = node_i.next_node
        self.list_storage_energy.append(node_i.energy())

        print(f"Done in {round(time.time()-start_time)} seconds.\n")
        return
    
    def print(self):
        node_i = self.source_node
        print(node_i)
        while node_i.next_node is not None:
            node_i = node_i.next_node
            print(node_i)

    def plot(self):
        # print("")
        # print(self.list_hp_energy)
        # print("")
        min_energy = Node(0,MIN_TOP_TEMP,1).energy()
        max_energy = Node(0,MAX_TOP_TEMP,NUM_LAYERS).energy()
        soc_list = [(x-min_energy)/(max_energy-min_energy)*100 for x in self.list_storage_energy]
        time_list = list(range(len(soc_list)))
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
        ax[0].step(time_list, self.list_hp_energy+[self.list_hp_energy[-1]], where='post', color='tab:blue', label='HP', alpha=0.6)
        ax[0].step(time_list, self.list_load+[self.list_load[-1]], where='post', color='tab:red', label='Load', alpha=0.6)
        ax[0].set_ylabel('Heat [kWh]')
        ax[0].set_ylim([-1,20])
        ax[0].legend(loc='upper left')
        ax2 = ax[0].twinx()
        ax2.step(time_list, self.list_elec_prices+[self.list_elec_prices[-1]], where='post', color='gray', alpha=0.6, label='Elec price')
        ax2.set_ylabel('Electricity price [cts/kWh]')
        ax2.legend(loc='upper right')
        ax[1].plot(time_list, soc_list, color='tab:orange', alpha=0.6, label='SoC')
        #ax3 = ax[1].twinx()
        #ax3.plot(time_list, self.list_thermoclines+[self.list_thermoclines[-1]], color='tab:green', alpha=0.5, label='Thermocline')
        #ax3.legend(loc='upper right')
        ax[1].legend(loc='upper left')
        ax[1].set_ylim([-1,101])
        ax[1].set_ylabel('SOC [%]')
        ax[1].set_xlabel('Time [hours]')
        if len(time_list)<50 and len(time_list)>10:
            ax[1].set_xticks(list(range(0,len(time_list)+1,2)))
        plt.show()


g = Graph(current_state=[0,50,6])
g.solve_dijkstra()
g.print()
g.plot()