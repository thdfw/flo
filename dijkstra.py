import time
import numpy as np
import pandas as pd
import pendulum
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from openpyxl.styles import PatternFill
from utils import COP, to_celcius, to_fahrenheit, get_data, required_SWT
from utils import (HORIZON_HOURS, MIN_TOP_TEMP_F, MAX_TOP_TEMP_F, TEMP_LIFT_F, NUM_LAYERS, 
                   MAX_HP_POWER_KW, MIN_HP_POWER_KW, STORAGE_VOLUME_GALLONS, LOSSES_PERCENT, 
                   CONSTANT_COP, TOU_RATES, START_TIME, START_TOP_TEMP_F, START_THERMOCLINE)


class Node():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        self.energy = self.get_energy()
        self.pathcost = 0 if time_slice==HORIZON_HOURS else 1e9
        self.next_node = None
        self.index=None

    def __repr__(self):
        return f"Node[time_slice:{self.time_slice}, top_temp:{self.top_temp}, thermocline:{self.thermocline}]"

    def get_energy(self):
        m_layer_kg = STORAGE_VOLUME_GALLONS * 3.785 / NUM_LAYERS
        energy_above_kWh = self.thermocline*m_layer_kg * 4.187/3600 * to_celcius(self.top_temp)
        energy_below_kWh = (NUM_LAYERS-self.thermocline)*m_layer_kg * 4.187/3600 * to_celcius(self.top_temp - TEMP_LIFT_F)
        return energy_above_kWh + energy_below_kWh


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
        print(f"Done in {round(time.time()-timer,2)} seconds.\n")

    def get_forecasts(self):
        df = get_data(self.start_time, HORIZON_HOURS)
        if TOU_RATES == 'old':
            self.elec_prices = list(df.jan24_prices)
        elif TOU_RATES == 'new':
            self.elec_prices = list(df.jul24_prices)
        self.oat = list(df.oat)
        self.load = list(df.load)
        self.min_SWT = [required_SWT(x) for x in self.load]

    def create_nodes(self):
        self.nodes = {}
        for time_slice in range(HORIZON_HOURS+1):
            self.nodes[time_slice] = [self.source_node] if time_slice==0 else []
            self.nodes[time_slice].extend(
                Node(time_slice, top_temp, thermocline)
                for top_temp in range(MIN_TOP_TEMP_F, MAX_TOP_TEMP_F+TEMP_LIFT_F, TEMP_LIFT_F) 
                for thermocline in list(range(NUM_LAYERS+1))
                if (time_slice, top_temp, thermocline) != (0, self.source_node.top_temp, self.source_node.thermocline)
            )
            nodes_by_energy = sorted(self.nodes[time_slice], key=lambda x: x.energy, reverse=True)
            for n in self.nodes[time_slice]:
                n.index = nodes_by_energy.index(n)+1
    
    def create_edges(self):
        self.edges = {}
        min_energy = Node(0,MIN_TOP_TEMP_F,0).energy
        energy_between_consecutive_states = Node(0,MIN_TOP_TEMP_F,2).energy - Node(0,MIN_TOP_TEMP_F,1).energy
        for h in range(HORIZON_HOURS):
            for node_now in self.nodes[h]:
                self.edges[node_now] = []
                for node_next in self.nodes[h+1]:
                    heat_to_store = node_next.energy - node_now.energy
                    losses = LOSSES_PERCENT/100*(node_now.energy-min_energy)
                    if losses<energy_between_consecutive_states and losses>0 and self.load[h]==0:
                        losses = energy_between_consecutive_states
                    heat_output_HP = heat_to_store + self.load[h] + losses
                    if heat_output_HP <= MAX_HP_POWER_KW and heat_output_HP >= MIN_HP_POWER_KW:
                        cop = COP(oat=self.oat[h], lwt=to_celcius(node_next.top_temp)) if CONSTANT_COP==0 else CONSTANT_COP
                        cost = self.elec_prices[h]/100 * heat_output_HP / cop
                        if heat_to_store > 0:
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline>node_now.thermocline:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                            if node_next.top_temp==node_now.top_temp+TEMP_LIFT_F:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                        elif heat_to_store < 0:
                            if self.load[h]>0 and (node_now.top_temp < self.min_SWT[h] or node_next.top_temp < self.min_SWT[h]):
                                continue
                            if node_next.top_temp==node_now.top_temp and node_next.thermocline<node_now.thermocline:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                            if node_next.top_temp==node_now.top_temp-TEMP_LIFT_F:
                                self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))
                        else:
                            self.edges[node_now].append(Edge(node_now, node_next, cost, heat_output_HP))

    def solve_dijkstra(self):
        print("Solving Dijkstra...")
        start_time = time.time()
        for time_slice in range(HORIZON_HOURS-1, -1, -1):
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
        max_edge_elec = max_edge.cost/self.elec_prices[0]*100
        min_edge_elec = min_edge.cost/self.elec_prices[0]*100
        bid = (min_edge.head.pathcost - max_edge.head.pathcost) / (max_edge_elec - min_edge_elec)
        if min_edge.head.top_temp==MIN_TOP_TEMP_F and min_edge.head.thermocline==1 and min_edge.heat_output_HP>1:
            print(f"Warning: The house will go cold if we don't buy now (minimum heat: {round(min_edge.heat_output_HP,1)} kWh)")
        self.bid = bid
        print(f"Bid price: {round(bid*100,2)} cts/kWh\n")

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
        min_energy = Node(0,MIN_TOP_TEMP_F,0).energy
        max_energy = Node(0,MAX_TOP_TEMP_F,NUM_LAYERS).energy
        list_soc = [(x-min_energy)/(max_energy-min_energy)*100 for x in self.list_storage_energy]
        # Plot the shortest path
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
        begin = self.start_time.format('YYYY-MM-DD HH:mm')
        end = self.start_time.add(hours=HORIZON_HOURS).format('YYYY-MM-DD HH:mm')
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
        norm = Normalize(vmin=MIN_TOP_TEMP_F-TEMP_LIFT_F, vmax=MAX_TOP_TEMP_F)
        cmap = matplotlib.colormaps['Reds']
        tank_top_colors = [cmap(norm(x)) for x in self.list_toptemps]
        tank_bottom_colors = [cmap(norm(x-TEMP_LIFT_F)) for x in self.list_toptemps]
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
        boundaries = np.arange(MIN_TOP_TEMP_F-TEMP_LIFT_F*1.5, MAX_TOP_TEMP_F+TEMP_LIFT_F*1.5, TEMP_LIFT_F)
        colors = [plt.cm.Reds(i/(len(boundaries)-1)) for i in range(len(boundaries))]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.025, pad=0.15, alpha=0.7)
        cbar.set_ticks(range(MIN_TOP_TEMP_F-TEMP_LIFT_F, MAX_TOP_TEMP_F+TEMP_LIFT_F, TEMP_LIFT_F))
        cbar.set_label('Temperature [F]')
        plt.show()

    def export_excel(self):
        warnings.filterwarnings("ignore")
        # First dataframe: the Dijkstra graph
        df = pd.DataFrame()
        nodes_by_energy = sorted(self.nodes[0], key=lambda x: (x.energy, x.top_temp), reverse=True)
        df['Top Temp [F]'] = [x.top_temp for x in nodes_by_energy]
        df['Thermocline'] = [x.thermocline for x in nodes_by_energy]
        df['Index'] = list(range(1,len(df)+1))
        for h in range(HORIZON_HOURS):
            df[h] = [[x.next_node.index, round(x.pathcost,2)] for x in sorted(self.nodes[h], key=lambda node: node.index, reverse=True)]
        df[HORIZON_HOURS] = [[0,0] for x in g.nodes[HORIZON_HOURS]]
        # Second dataframe: the forecasts
        df2 = pd.DataFrame({'Forecast':['0'], **{h: [0.0] for h in range(HORIZON_HOURS+1)}})
        df2.loc[0] = ['Price [cts/kWh]'] + g.elec_prices
        df2.loc[1] = ['Load [kW]'] + g.load
        df2.loc[2] = ['OAT [F]'] +[round(to_fahrenheit(x)) for x in g.oat]
        df2.loc[3] = ['Required SWT [F]'] +[round(x) for x in g.min_SWT]
        df2.reset_index(inplace=True, drop=True)
        # Highlight shortest path
        highlight_positions = []
        node_i = g.source_node
        while node_i.next_node is not None:
            highlight_positions.append((node_i.index+len(df2)+2, 3+node_i.time_slice))
            node_i = node_i.next_node
        highlight_positions.append((node_i.index+len(df2)+2, 3+node_i.time_slice))
        # Export excel
        with pd.ExcelWriter('dijkstra_result.xlsx', engine='openpyxl') as writer:
            df2.to_excel(writer, index=False, startcol=2, sheet_name='Sheet1')
            df.to_excel(writer, index=False, startrow=len(df2)+2, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']
            worksheet.column_dimensions['A'].width = 15
            worksheet.column_dimensions['B'].width = 15
            worksheet.column_dimensions['C'].width = 15
            highlight_fill = PatternFill(start_color='72ba93', end_color='72ba93', fill_type='solid')
            for row, col in highlight_positions:
                worksheet.cell(row=row+1, column=col+1).fill = highlight_fill


if __name__ == '__main__':
    
    time_now = START_TIME
    state_now = Node(time_slice=0, top_temp=START_TOP_TEMP_F, thermocline=START_THERMOCLINE)

    g = Graph(state_now, time_now)
    g.solve_dijkstra()
    g.export_excel()
    g.compute_bid()
    g.plot()