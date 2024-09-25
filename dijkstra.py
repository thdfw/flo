import time
import matplotlib.pyplot as plt

HORIZON = 48 # hours
HP_POWER = 12 # kW
M_LAYER = 113 # kg
MIN_TOP_TEMP = 50 # C
MAX_TOP_TEMP = 85 # C
TEMP_LIFT = 11.11 # C
NUM_LAYERS = 12
PRINT = False


class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        # Initial values for Dijkstra
        self.dist = 1e9
        self.prev = None

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


class Graph():
    def __init__(self, current_state):
        self.get_forecasts()
        self.define_nodes(current_state)
        self.define_edges()
        self.list_hp_energy = []
        self.list_storage_energy = []
        self.list_elec_prices = []
        self.list_load = []

    def get_forecasts(self):
        self.elec_prices = [7.92, 6.63, 6.31, 6.79, 8.01, 11.58, 19.38, 21.59, 11.08, 4.49, 1.52, 
                           0.74, 0.42, 0.71, 0.97, 2.45, 3.79, 9.56, 20.51, 28.26, 23.49, 18.42, 13.23, 10.17]*3
        self.oat = [-2]*HORIZON
        self.load = [4]*HORIZON

    def COP(self, oat, ewt, lwt):
        return 2

    def define_nodes(self, current_state):
        time_slice0, top_temp0, thermocline0 = current_state
        self.nodes = [Node(time_slice0, top_temp0, thermocline0)]
        self.nodes.extend([
            Node(time_slice, top_temp, 6) 
            for time_slice in range(HORIZON) 
            for top_temp in range(MIN_TOP_TEMP, MAX_TOP_TEMP) 
            #for thermocline in range(1, NUM_LAYERS+1)
            #if (time_slice, top_temp, thermocline) != (time_slice0, top_temp0, thermocline0)
            ])
    
    def define_edges(self):
        self.edges = []
        for time in range(HORIZON):
            for node1 in [x for x in self.nodes if x.time_slice==time]:
                for node2 in [x for x in self.nodes if x.time_slice==time+1]:

                    energy_to_store = node2.energy() - node1.energy()
                    energy_from_HP = energy_to_store + self.load[node1.time_slice]

                    if energy_from_HP <= HP_POWER and energy_from_HP >= 0:
                        cop = self.COP(oat=self.oat[node1.time_slice], ewt=node1.top_temp-TEMP_LIFT, lwt=node2.top_temp)
                        elec_cost = self.elec_prices[node1.time_slice]
                        cost = elec_cost * energy_from_HP / cop 
                        self.edges.append(Edge(node2,node1,cost))

    def solve_dijkstra(self):
        source_node = self.nodes[0]
        source_node.dist = 0
        self.nodes_unvisited = self.nodes.copy()
        previous_percentage = -5
        while self.nodes_unvisited:
            # Find the unvisited node which is closest to the source
            closest_node = min(self.nodes_unvisited, key=lambda x: x.dist)
            if closest_node.dist == 1e9:
                break
            # For all unvisisted connected nodes, find their distance to the source through the current node
            self.neighbours = [x.tail for x in self.edges if (x.head==closest_node and x.tail in self.nodes_unvisited)]
            for n in self.neighbours:
                e = [x for x in self.edges if x.tail==n and x.head==closest_node][0]
                distance = closest_node.dist + e.cost
                if distance < n.dist:
                    n.dist = distance
                    n.prev = closest_node
            # Mark the current node as visited
            self.nodes_unvisited.remove(closest_node)
            # Print progress
            percentage_progress = 100 - round(len(self.nodes_unvisited) / len(self.nodes) * 100)
            if percentage_progress >= previous_percentage+5:
                print(f"{percentage_progress}%...")
                previous_percentage = percentage_progress
        # Go through the shortest path
        min_node = min((x for x in self.nodes if x.time_slice==HORIZON-1), key=lambda n: n.dist)
        shortest_path = [self.nodes.index(min_node)]
        while min_node != source_node:
            energy_to_store = min_node.energy() - min_node.prev.energy()
            energy_from_HP = energy_to_store + self.load[min_node.prev.time_slice]
            if PRINT:
                print(f"---- Time {min_node.prev.time_slice} to {min_node.time_slice} ----")
                print(f"Elec price: {self.elec_prices[min_node.prev.time_slice]}")
                print(f"Top temperature: {min_node.prev.top_temp} -> {min_node.top_temp}")
                print(f"Thermocline: {min_node.prev.thermocline} -> {min_node.thermocline}")
                print(f"Energy from HP: {round(energy_from_HP,2)}\n")
            self.list_storage_energy = [min_node.energy()] + self.list_storage_energy
            self.list_elec_prices = [self.elec_prices[min_node.prev.time_slice]] + self.list_elec_prices
            self.list_load = [self.load[min_node.prev.time_slice]] + self.list_load
            self.list_hp_energy = [round(energy_from_HP,2)] + self.list_hp_energy
            min_node = min_node.prev
            shortest_path.append(self.nodes.index(min_node))
        # Return the next node to visit
        return self.nodes[shortest_path[-2]]
    
    def plot(self):
        min_energy = Node(0,MIN_TOP_TEMP,1).energy()
        max_energy = Node(0,MAX_TOP_TEMP,NUM_LAYERS).energy()
        initial_energy = self.nodes[0].energy()
        list_storage_energy = [initial_energy] + self.list_storage_energy
        soc_list = [(x-min_energy)/(max_energy-min_energy)*100 for x in list_storage_energy]
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
        ax[1].plot(time_list, soc_list, color='tab:orange', alpha=0.6)
        ax[1].set_ylabel('SOC [%]')
        ax[1].set_xlabel('Time [hours]')
        if len(time_list)<50 and len(time_list)>10:
            ax[1].set_xticks(list(range(0,len(time_list)+1,2)))
        plt.show()


g = Graph(current_state=[0,51,1])
start_time = time.time()
next_node = g.solve_dijkstra()
print(f"Dijkstra ran in {round(time.time()-start_time,2)} seconds.")
g.plot()