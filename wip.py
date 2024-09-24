# type: ignore
import math
import random
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import no_type_check

import numpy as np
import pendulum

import gwatn.brick_storage_heater.strategy_utils as strategy_utils
from gwatn.brick_storage_heater.edge import Edge__BrickStorageHeater as Edge
from gwatn.brick_storage_heater.node import Node_BrickStorageHeater as Node
from gwatn.data_classes.d_graph import DGraph
from gwatn.types import FloParamsBrickstorageheater as FloParams


class Flo__BrickStorageHeater(DGraph):
    MAGIC_HEAT_PUMP_DELTA_F = 20
    FAILED_HEATING_PENALTY_DOLLARS = 10**6

    def __init__(
        self,
        params: FloParams,
        d_graph_id: str,
    ):
        self.params = params

        self.RealtimeElectricityPrice = np.array(self.params.RealtimeElectricityPrice)
        self.DistributionPrice = np.array(self.params.DistributionPrice)
        if self.params.IsRegulating:
            self.reg_price_per_mwh = np.array(self.params.RegulationPrice)
        else:
            self.reg_price_per_mwh = np.array(
                [0] * len(self.params.RealtimeElectricityPrice)
            )

        self.max_energy_kwh_th = strategy_utils.get_max_energy_kwh(self.params)

        self.currency_unit = self.params.CurrencyUnit
        self.temp_unit = self.params.TempUnit
        DGraph.__init__(
            self,
            d_graph_id=d_graph_id,
            graph_strategy_alias="SpaceHeat__HeatPumpWithBoostStore__Flo",
            flo_start_unix_time_s=pendulum.datetime(
                year=self.params.StartYearUtc,
                month=self.params.StartMonthUtc,
                day=self.params.StartDayUtc,
                hour=self.params.StartHourUtc,
                minute=self.params.StartMinuteUtc,
            ).timestamp(),
            slice_duration_hrs=list(np.array(self.params.SliceDurationMinutes) / 60),
            default_storage_steps=self.params.StorageSteps,
            starting_store_idx=self.params.StartingStoreIdx,
            timezone_string=self.params.TimezoneString,
            home_city=self.params.HomeCity,
            currency_unit=self.params.CurrencyUnit,
            max_storage=self.max_energy_kwh_th,
            max_power_in=self.params.RatedMaxPowerKw,
            wh_exponent=3,
        )
        self.e_step: float = self.max_energy_kwh_th / self.params.StorageSteps
        room_temp_c = (self.params.RoomTempF - 32) * 5 / 9
        temp_range_c = self.params.MaxBrickTempC - room_temp_c
        self.temp_step_c: float = temp_range_c / self.params.StorageSteps
        self.store_kwh_per_deg_c = self.max_energy_kwh_th / temp_range_c
        self.energy_cost_per_kwh: Dict[int, float] = {}
        self.set_energy_cost_per_kwh()
        self.uncosted_edges: Dict[Edge, int] = {}
        self.failed_in_cost_boost_preferred: int = 0
        self.failed_in_cost_hp_preferred: int = 0

        self.create_graph()
        self.solve_dijkstra()

    def create_slice_nodes(self, ts_idx: int) -> None:
        """Creates nodes for time slice ts_idx, equally spaced by self.e_step,
        running from params.ZeroPotentialEnergyWaterTempF up to params.MaxStoreTempF.

        Sets the energy store in kwh of enthalpy (latent heat in this case) as well
        as the average boost water temp.

        """
        ...

    def create_nodes(self) -> None:
        for ts_idx in range(self.time_slices + 1):
            self.create_slice_nodes(ts_idx)

    def set_energy_cost_per_kwh(self):
        for ts_idx in range(self.time_slices):
            self.energy_cost_per_kwh[ts_idx] = self.get_energy_cost_per_kwh(ts_idx)

    ######################################################
    # Functions of edges
    ######################################################

    # Cost related

    def get_energy_cost_per_kwh(self, ts_idx: int) -> float:
        energy_cost_per_mwh = self.params.RealtimeElectricityPrice[ts_idx]
        dist_cost_per_mwh = self.params.DistributionPrice[ts_idx]

        cost_per_mwh = energy_cost_per_mwh + dist_cost_per_mwh
        return cost_per_mwh / 1000

    ######################################################
    # Functions of edges - Water temp related
    #####################################################

    def set_failed_edge_properties(self, edge: Edge):
        """Set costs"""
        edge.cost = self.FAILED_HEATING_PENALTY_DOLLARS
        edge.hp_electricity_avg_kw = 10000
        edge.avg_kw = 0

    def get_failed_edge(self, node: Node) -> Edge:
        edge = Edge(
            start_ts_idx=node.ts_idx, start_idx=node.store_idx, end_idx=node.store_idx
        )
        self.set_failed_edge_properties(edge)
        return edge

    def get_uncosted_edge(
        self, node: Node, delta_energy_kwh: float, existing_edges: List[Edge] = []
    ) -> Optional[Edge]:
        """Finds a randomized passing edge consistent with delta_energy_kwh.
        For example, if delta_energy_kwh falls exactly half way between
        two energy steps it will return each step with probability 50%
        """
        if node.store_enthalpy_kwh + delta_energy_kwh < 0:
            return None
        delta_idx_as_float = delta_energy_kwh / self.e_step
        lower_delta_idx = math.floor(delta_idx_as_float)
        frac = delta_idx_as_float - lower_delta_idx
        random_adder = random.choices(population=[0, 1], weights=[1 - frac, frac], k=1)[
            0
        ]
        if node.store_idx + lower_delta_idx == self.params.StorageSteps:
            delta_idx = lower_delta_idx
        else:
            delta_idx = lower_delta_idx + random_adder
        edge = Edge(
            start_ts_idx=node.ts_idx,
            start_idx=node.store_idx,
            end_idx=node.store_idx + delta_idx,
        )
        if edge not in existing_edges:
            self.uncosted_edges[edge] = 1
            return edge
        else:
            # in this case the edge is already in the existing_edges, so return None
            return None

    def get_edges_from_node(self, node: Node) -> List[Edge]:
        """Creates edges starting at a node."""
        ts_idx = node.ts_idx
        return []

    def set_edge_cost(self, edge: Edge) -> None:
        """Given an edge:
        - set cost to penalty if there is no way to meet the energy requirements
        - otherwise set the cost for the optimal choice of using the boost and
        the heat pump
        """
        ...

    def create_graph(self) -> None:
        # print(f"Creating graph nodes, edges and edge weights")
        st = time.time()
        self.create_nodes()
        nt = time.time()
        # print(f"{time.time() - st:1.0f} seconds to make nodes")
        for jj in range(self.time_slices):
            # if jj % 500 == 0:
            #     print(f"{time.time() - st:1.0f} seconds for {jj} slices")
            for node in self.node[jj].values():
                edges = self.get_edges_from_node(node=node)
                self.edges[node] = edges
        et = time.time()
        # print(f"{et - nt:1.0f} seconds for building edges")
        for edge in self.uncosted_edges.keys():
            self.set_edge_cost(edge)
        self.uncosted_edges = {}
        ct = time.time()
        # print(f"{ct - et:1.0f} seconds for costing edges")
        tt = time.time() - st
        # print(f"time per 100 slices:{tt*100000/self.time_slices:2.0f} ms ")
        # print(f"total time to build graph: {tt:2.0f} s")
        # print(f"Total number of slices: {self.time_slices}")