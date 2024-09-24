from typing import List

class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline

        self.distance = 1e9


class Edge():
    def __init__(self, tail:Node, head:Node, cost:float):
        self.tail = tail
        self.head = head
        self.cost = cost


class Graph():
    def __init__(self, nodes=List[Node], edges=List[Edge]):
        self.nodes = nodes
        self.edges = edges

    def print_graph(self):
        print(f"Nodes: n0 to n{len(self.nodes)-1}")
        print("Edges:")
        for e in edges:
            print(f"n{nodes.index(e.tail)}-n{nodes.index(e.head)}, cost {e.cost}")

    def solve_dijkstra(self, target_node:Node):

        print(f"Finding the shortest path to v{self.nodes.index(target_node)}...\n")

        # Initialize
        target_node.distance = 0
        self.nodes_unvisited = self.nodes.copy()

        while self.nodes_unvisited:

            # Find the unvisited node which is closest to the target
            closest_node = min(self.nodes_unvisited, key=lambda x: x.distance)
            if closest_node.distance == 1e9:
                break
            print(f"{'*'*40}\nCurrent node: v{self.nodes.index(closest_node)}\n{'*'*40}")

            # For all unvisisted connected nodes, find their distance to the target through the current node
            self.neighbours = [x.tail for x in self.edges if (x.head==closest_node and x.tail in self.nodes_unvisited)]
            for n in self.neighbours:
                print(f"Checking neighbour: n{self.nodes.index(n)}")
                e = [x for x in self.edges if x.tail==n and x.head==closest_node][0]
                if closest_node.distance + e.cost < n.distance:
                    n.distance = closest_node.distance + e.cost

            # Mark the current node as visited
            self.nodes_unvisited.remove(closest_node)

        for n in self.nodes:
            print(f'n{self.nodes.index(n)}')
            print(n.distance)



nodes = []
nodes.append(Node(0,0,0))
nodes.append(Node(3,0,0))
nodes.append(Node(2,0,0))
nodes.append(Node(1,0,0))
nodes.append(Node(3,0,0))
nodes.append(Node(2,0,0))
nodes.append(Node(1,0,0))

edges = []
edges.append(Edge(nodes[1],nodes[2],1))
edges.append(Edge(nodes[1],nodes[5],2))
edges.append(Edge(nodes[2],nodes[3],-3))
edges.append(Edge(nodes[2],nodes[6],1))
edges.append(Edge(nodes[3],nodes[0],1))
edges.append(Edge(nodes[4],nodes[5],5))
edges.append(Edge(nodes[5],nodes[6],0))
edges.append(Edge(nodes[6],nodes[0],0))

g = Graph(nodes,edges)
# g.print_graph()
g.solve_dijkstra(nodes[0])