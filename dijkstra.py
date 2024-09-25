from typing import List, Optional

HORIZON = 15 # hours
HP_POWER = 12 # kW
M_LAYER = 113 # kg


class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        # Initial values for Dijkstra
        self.dist: float = 1e9
        self.prev: Optional[Node] = None

    def energy(self):
        m_top = (self.thermocline-1)*M_LAYER
        m_bottom = (12-self.thermocline)*M_LAYER
        joules = m_top*4187*self.top_temp + m_bottom*4187*(self.top_temp-11.11)
        kWh = joules/3600/1000
        return round(kWh,2)


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

    def get_forecasts(self):
        self.elec_costs = [0]*4 + [1e5]*100
        self.oat = [-2]*HORIZON
        self.load = [5]*HORIZON

    def COP(self, oat, ewt, lwt):
        return 2

    def define_nodes(self, current_state):
        time_slice0, top_temp0, thermocline0 = current_state
        self.nodes = [Node(time_slice0, top_temp0, thermocline0)]
        self.nodes.extend([
            Node(time_slice, top_temp, thermocline) 
            for time_slice in range(HORIZON) 
            for top_temp in range(59, 80) 
            for thermocline in range(1, 3)
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
                        cop = self.COP(oat=self.oat[node1.time_slice], ewt=0, lwt=0)
                        cost = self.elec_costs[node1.time_slice] / cop * energy_from_HP if energy_from_HP>0 else 0
                        self.edges.append(Edge(node2,node1,cost))

    def solve_dijkstra(self):
        source_node = self.nodes[0]
        source_node.dist = 0
        self.nodes_unvisited = self.nodes.copy()
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
        # Find the node in the final layer which has the lowest cost
        min_node = min((x for x in self.nodes if x.time_slice==HORIZON-1), key=lambda n: n.dist)
        shortest_path = [self.nodes.index(min_node)]
        while min_node != source_node:
            print(f"---- Time {min_node.prev.time_slice} to {min_node.time_slice} ----")
            print(f"Elec price: {self.elec_costs[min_node.prev.time_slice]}")
            print(f"Top temperature: {min_node.prev.top_temp} -> {min_node.top_temp}")
            print(f"Thermocline: {min_node.prev.thermocline} -> {min_node.thermocline}")
            print(f"Energy: {min_node.prev.energy()} -> {min_node.energy()}\n")
            min_node = min_node.prev
            shortest_path.append(self.nodes.index(min_node))
        # Return the next node to visit
        return self.nodes[shortest_path[-2]]


g = Graph(current_state=[0,60,3])
next_node = g.solve_dijkstra()