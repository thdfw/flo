from typing import List, Optional


class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        # Initial values for Dijkstra
        self.dist: float = 1e9
        self.prev: Optional[Node] = None


class Edge():
    def __init__(self, tail:Node, head:Node, cost:float):
        self.tail = tail
        self.head = head
        self.cost = cost


class Graph():
    def __init__(self, nodes=List[Node], edges=List[Edge]):
        self.nodes = nodes
        self.edges = edges
        self.time_slices = max((x.time_slice for x in nodes))

    def solve_dijkstra(self, source_node:Node):
        self.nodes_unvisited = self.nodes.copy()
        source_node.dist = 0
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
        min_node = min((x for x in self.nodes if x.time_slice==self.time_slices), key=lambda n: n.dist)
        shortest_path = [self.nodes.index(min_node)]
        while min_node != source_node:
            min_node = min_node.prev
            shortest_path.append(self.nodes.index(min_node))
        # Return the next node to visit
        return self.nodes[shortest_path[-2]]




# Define all the nodes
nodes = []
for t in range(48):
    for x1 in range(30,60):
        for x2 in range(1,12):
            print(f"{t},{x1},{x2}")
            nodes.append(Node(t,x1,x2))

# Define all the edges
edges = []
for n1 in nodes:
    for n2 in  [x for x in nodes if x.time_slice==n1.time_slice-1]:
        # Check if (n1,n2) can exist
        print('')
        # Assing a cost

# Define the graph
g = Graph(nodes,edges)


# if __name__ == '__main__':

#     nodes = []
#     nodes.append(Node(0,0,0))
#     nodes.append(Node(3,0,0))
#     nodes.append(Node(2,0,0))
#     nodes.append(Node(1,0,0))
#     nodes.append(Node(3,0,0))
#     nodes.append(Node(2,0,0))
#     nodes.append(Node(1,0,0))

#     edges = []
#     edges.append(Edge(nodes[1],nodes[2],1))
#     edges.append(Edge(nodes[1],nodes[5],2))
#     edges.append(Edge(nodes[2],nodes[3],3))
#     edges.append(Edge(nodes[2],nodes[6],1))
#     edges.append(Edge(nodes[3],nodes[0],1))
#     edges.append(Edge(nodes[4],nodes[5],5))
#     edges.append(Edge(nodes[5],nodes[6],0))
#     edges.append(Edge(nodes[6],nodes[0],0))

#     g = Graph(nodes,edges)
#     next_node = g.solve_dijkstra(nodes[0])