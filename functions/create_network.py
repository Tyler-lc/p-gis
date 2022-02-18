#######About##########################
# Author: Bernhard Felber, TU WIEN, Energy Economics Group
# Date: 16.07.21
# Project: Emb3rs

################################################
################Packages to load################
################################################

from ctypes import Union
from typing import Dict
from black import out
import jsonpickle
import osmnx as ox
import networkx as nx
from osmnx.distance import euclidean_dist_vec
import pandas as pd
from pyomo.environ import *
from pyomo.opt import *
import haversine as hs
from shapely.geometry import Polygon, Point
import sklearn
from networkx.readwrite import json_graph

from ..utilities.create_ex_grid import create_ex_grid

################################################
################Create Network  ################
################################################


def create_network(
    n_supply_dict,
    n_demand_dict,
    ex_cap={},
    ex_grid=nx.MultiDiGraph(),
    network_resolution="high",
    coords_list=[],
):
    polygon = Polygon(coords_list).convex_hull

    ex_cap = pd.DataFrame(ex_cap)
    # readinf ex_cap from json makes all column names str
    # convert the datatype of columns names (only time steps) to int from str
    ex_cap_cols = ex_cap.columns.values
    ex_cap_cols[3:] = ex_cap_cols[3:].astype(int)
    ex_cap.columns = ex_cap_cols

    if len(ex_cap) == 0:
        pass
    else:
        ex_cap.iloc[:, 3 : len(ex_cap.columns)] = (
            ex_cap.iloc[:, 3 : len(ex_cap.columns)] / 1000
        )
        ################################################################################
        ######ITERATION TEO - DELETE SOURCES/SINKS THAT WERE EXCLUDED FROM TEO##########

        Source_list_TEO = ex_cap.copy()[ex_cap["classification_type"] == "source"]
        Source_list_TEO["sum"] = Source_list_TEO.iloc[:, 3:-1].sum(
            axis=1
        )  #####hier weitermachen
        Source_list_TEO = Source_list_TEO[Source_list_TEO["sum"] > 0].drop(
            "sum", axis=1
        )
        Source_list_TEO = list(Source_list_TEO["number"])

        Sources_to_delete = set(list(n_supply_dict)) - set(list(Source_list_TEO))

        if len(Source_list_TEO) > 0:
            for i in Sources_to_delete:
                del n_supply_dict[i]

        Sink_list_TEO = ex_cap.copy()[ex_cap["classification_type"] == "sink"]
        Sink_list_TEO["sum"] = Sink_list_TEO.iloc[:, 3:-1].sum(axis=1)
        Sink_list_TEO = Sink_list_TEO[Sink_list_TEO["sum"] > 0]
        Sink_list_TEO = list(Sink_list_TEO["number"])

        Sinks_to_delete = set(list(n_demand_dict)) - set(list(Sink_list_TEO))

        if len(Sink_list_TEO) > 0:
            for i in Sinks_to_delete:
                del n_demand_dict[i]

    ################################################################################
    ######GET OSM ROAD NETWORK######################################################

    if network_resolution == "high":

        cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential|living_street|service|pedestrian|unclassified|track|road|path"]'

        road_nw = ox.graph_from_polygon(
            polygon, simplify=False, clean_periphery=True, custom_filter=cf
        )
        road_nw = ox.simplify_graph(road_nw)

    elif network_resolution == "low":

        cf = '["highway"~"primary|primary_link|secondary|secondary_link"]'

        road_nw = ox.graph_from_polygon(
            polygon, simplify=False, clean_periphery=True, custom_filter=cf
        )
        road_nw = ox.simplify_graph(road_nw)
    else:
        pass

    ######## Pass osmid to nodes
    for node in road_nw.nodes:
        road_nw.nodes[node]["osmid"] = node

    ###########REMOVE LONGER EDGES BETWEEN POINTS IF MUTIPLE EXIST#######################
    ##AS ONLY ONE (u,v - v,u) EDGE BETWEEN TWO POINTS CAN BE CONSIDERED FOR OPTIMIZATION#

    nodes, edges = ox.graph_to_gdfs(road_nw)

    network_edges_crs = edges.crs
    network_nodes_crs = nodes.crs

    edges_to_drop = []
    edges = edges.reset_index(level=[0, 1, 2])
    edges_double = pd.DataFrame(edges)
    edges_double["id"] = (
        edges_double["u"].astype(str) + "-" + edges_double["v"].astype(str)
    )

    for i in edges_double["id"].unique():
        double_edges = edges_double[edges_double["id"] == i]
        if len(double_edges) > 1:
            mx_ind = double_edges["length"].idxmin()
            mx = double_edges.drop(mx_ind)
            edges_to_drop.append(mx)
        else:
            None

    try:
        edges_to_drop = pd.concat(edges_to_drop)
        for i in zip(edges_to_drop["u"], edges_to_drop["v"], edges_to_drop["key"]):
            road_nw.remove_edge(u=i[0], v=i[1], key=i[2])
    except:
        None

    #########################REMOVE LOOPS FOR ONE WAYS#######################
    ######HAPPENS IF THE TWO EDGES BETWEEN TWO POINTS DO NOT HAVE THE########
    ###########################SAME LENGTH###################################

    nodes, edges = ox.graph_to_gdfs(road_nw)
    edges = edges.reset_index(level=[0, 1, 2])

    edges_one_way = pd.DataFrame(edges[edges["oneway"] == True])
    edges_one_way["id"] = list(zip(edges_one_way["u"], edges_one_way["v"]))

    edges_to_drop = []

    for i in edges_one_way["id"]:
        edges_u_v = edges_one_way[
            (edges_one_way["u"] == i[0]) & (edges_one_way["v"] == i[1])
        ]
        edges_v_u = edges_one_way[
            (edges_one_way["u"] == i[1]) & (edges_one_way["v"] == i[0])
        ]
        edges_all = pd.concat([edges_u_v, edges_v_u])
        if len(edges_all) > 1:
            mx_ind = edges_all["length"].idxmin()
            mx = edges_all.drop(mx_ind)
            edges_to_drop.append(mx)
        else:
            None

    try:
        edges_to_drop = pd.concat(edges_to_drop).drop("id", axis=1)
        edges_to_drop = edges_to_drop[~edges_to_drop.index.duplicated(keep="first")]
        edges_to_drop = edges_to_drop.drop_duplicates(subset=["length"], keep="last")
        for i in zip(edges_to_drop["u"], edges_to_drop["v"], edges_to_drop["key"]):
            road_nw.remove_edge(u=i[0], v=i[1], key=i[2])
    except:
        None

    ################################################################################
    ######CREATE ATTRIBUTES TO BE POSSIBLY CHANGED BY THE USER######################

    nx.set_edge_attributes(road_nw, "street", "surface_type")
    nx.set_edge_attributes(road_nw, 0, "restriction")
    nx.set_edge_attributes(road_nw, 0, "surface_pipe")
    nx.set_edge_attributes(road_nw, 0, "existing_grid_element")
    nx.set_edge_attributes(road_nw, 0, "inner_diameter_existing_grid_element")
    nx.set_edge_attributes(road_nw, 0, "costs_existing_grid_element")

    ################################################################################
    ######CHECK IF EXISTING GRID ELEMENT HAS BEEN UPLOADED AS WELL##################

    if len(ex_grid) == 0:
        pass
    else:

        ################################################################################
        ######ALIGN CRS OF EXISTING GRID AND OSM GRAPH##################################

        nodes_ex_grid, edges_ex_grid = ox.graph_to_gdfs(ex_grid)
        nodes_ex_grid.crs = network_nodes_crs
        edges_ex_grid.crs = network_edges_crs
        ex_grid = ox.graph_from_gdfs(nodes_ex_grid, edges_ex_grid)

        ################################################################################
        ######FIND CLOSEST POINTS BETWEEN OSM GRAPH AND EX GRID#########################

        dist = sklearn.neighbors.DistanceMetric.get_metric("haversine")
        nodes_ex = ox.graph_to_gdfs(ex_grid, nodes=True, edges=False)
        nodes_rn = ox.graph_to_gdfs(road_nw, nodes=True, edges=False)

        dist_matrix = dist.pairwise(nodes_ex[["y", "x"]], nodes_rn[["y", "x"]])
        dist_matrix = pd.DataFrame(
            dist_matrix, index=nodes_ex.index, columns=nodes_rn.index
        )

        ################################################################################
        ######RETRIEVE INDEX OF MIN VALUE FROM DISTANCE MATRIX##########################

        col = dist_matrix.min().idxmin()
        ind = dist_matrix.loc[:, col].idxmin()

        ################################################################################
        ####ADD NECESSARY ATTRIBUTES TO EXISTING GRAPH OBJECT TO PERFORM CALCULATION####

        nx.set_edge_attributes(ex_grid, "None", "surface_type")
        nx.set_edge_attributes(ex_grid, 0, "restriction")
        # nx.set_edge_attributes(ex_grid, 0,"surface_pipe")  ###maybe this could possibly be already part of upload if we allow surface pipes
        nx.set_edge_attributes(ex_grid, 1, "existing_grid_element")
        # nx.set_edge_attributes(ex_grid, 0.5,"diameter_existing_grid_element")  ###this attribute already needs to come from the upload, maybe the name needs to be transformed
        # nx.set_edge_attributes(ex_grid, 0, "costs_existing_grid_element")  ###this attribute already needs to come from the upload, maybe the name needs to be transformed

        ################################################################################
        ######COMBINE BOTH GRAPHS AND ADD EDGE WITH DISTANCE ATTRIBUTE##################

        road_nw = nx.compose(ex_grid, road_nw)
        road_nw.add_edge(
            ind,
            col,
            length=hs.haversine(
                (road_nw.nodes[ind]["y"], road_nw.nodes[ind]["x"]),
                (road_nw.nodes[col]["y"], road_nw.nodes[col]["x"]),
            )
            * 1000,
            surface_type="street",
            restriction=0,
            surface_pipe=0,
            existing_grid_element=0,
            inner_diameter_existing_grid_element=0,
            costs_existing_grid_element=0,
        )

    road_nw_streets = road_nw.copy()

    ################################################################################
    ######CONNECT SOURCES AND SINKS TO OSM GRAPH####################################

    for k, v in n_supply_dict.items():
        dist_edge = ox.get_nearest_edge(
            road_nw_streets, (v["coords"][0], v["coords"][1])
        )
        dist_1 = euclidean_dist_vec(
            v["coords"][0],
            v["coords"][1],
            road_nw_streets.nodes[dist_edge[0]]["y"],
            road_nw_streets.nodes[dist_edge[0]]["x"],
        )
        dist_2 = euclidean_dist_vec(
            v["coords"][0],
            v["coords"][1],
            road_nw_streets.nodes[dist_edge[1]]["y"],
            road_nw_streets.nodes[dist_edge[1]]["x"],
        )
        dist_dict = {
            road_nw_streets.nodes[dist_edge[0]]["osmid"]: dist_1,
            road_nw_streets.nodes[dist_edge[1]]["osmid"]: dist_2,
        }
        point_to_connect = min(dist_dict, key=dist_dict.get)
        road_nw.add_node(k, y=v["coords"][0], x=v["coords"][1], osmid=k)
        road_nw.add_edge(
            k,
            point_to_connect,
            length=hs.haversine(
                (v["coords"][0], v["coords"][1]),
                (
                    road_nw.nodes[point_to_connect]["y"],
                    road_nw.nodes[point_to_connect]["x"],
                ),
            )
            * 1000,
            surface_type="street",
            restriction=0,
            surface_pipe=0,
            existing_grid_element=0,
            inner_diameter_existing_grid_element=0,
            costs_existing_grid_element=0,
        )

    for k, v in n_demand_dict.items():
        dist_edge = ox.get_nearest_edge(
            road_nw_streets, (v["coords"][0], v["coords"][1])
        )
        dist_1 = euclidean_dist_vec(
            v["coords"][0],
            v["coords"][1],
            road_nw_streets.nodes[dist_edge[0]]["y"],
            road_nw_streets.nodes[dist_edge[0]]["x"],
        )
        dist_2 = euclidean_dist_vec(
            v["coords"][0],
            v["coords"][1],
            road_nw_streets.nodes[dist_edge[1]]["y"],
            road_nw_streets.nodes[dist_edge[1]]["x"],
        )
        dist_dict = {
            road_nw_streets.nodes[dist_edge[0]]["osmid"]: dist_1,
            road_nw_streets.nodes[dist_edge[1]]["osmid"]: dist_2,
        }
        point_to_connect = min(dist_dict, key=dist_dict.get)
        road_nw.add_node(k, y=v["coords"][0], x=v["coords"][1], osmid=k)
        road_nw.add_edge(
            k,
            point_to_connect,
            length=hs.haversine(
                (v["coords"][0], v["coords"][1]),
                (
                    road_nw.nodes[point_to_connect]["y"],
                    road_nw.nodes[point_to_connect]["x"],
                ),
            )
            * 1000,
            surface_type="street",
            restriction=0,
            surface_pipe=0,
            existing_grid_element=0,
            inner_diameter_existing_grid_element=0,
            costs_existing_grid_element=0,
        )

    ################################################################################
    ####PROJECT GRAPH AND TURN INTO UNDIRECTED FOR USER DISPLAY#####################

    road_nw = ox.get_undirected(road_nw)
    road_nw = ox.project_graph(road_nw)

    ################################################################################
    ####################DELETE ATTRIBUTES FROM EDGES NOT NEEDED#####################

    nodes, edges = ox.graph_to_gdfs(road_nw)
    cols_to_drop = [
        "osmid",
        "oneway",
        "name",
        "highway",
        "maxspeed",
        "lanes",
        "junction",
        "service",
        "access",
    ]
    edges = edges.drop(edges.columns.intersection(cols_to_drop), axis=1)
    road_nw = ox.graph_from_gdfs(nodes, edges)

    # extract the nodes and edges from the graphs and convert them to GoeJSON
    nodes_json = nodes.to_dict("records")
    edges_json = edges.to_dict("records")

    return (
        nodes_json,
        edges_json,
        road_nw,
        n_demand_dict,
        n_supply_dict,
    )  # road_nw is given as output for test purposes


def run_create_network(input_data):
    (
        n_supply_list,
        n_demand_list,
        ex_grid,
        in_cap,
        network_resolution,
        coords_list,
    ) = prepare_input(input_data)

    nodes, edges, road_nw, n_demand_dict, n_supply_dict = create_network(
        n_supply_dict=n_supply_list,
        n_demand_dict=n_demand_list,
        ex_grid=ex_grid,
        ex_cap=in_cap,
        network_resolution=network_resolution,
        coords_list=coords_list,
    )

    return prepare_output(nodes, edges, road_nw, n_demand_dict, n_supply_dict)


## Prepare Input Data to Function
def prepare_input(input_data):
    n_supply_list = input_data["n_supply_list"]
    n_demand_list = input_data["n_demand_list"]
    ex_grid_data_json = input_data["ex_grid_data_json"]
    in_cap = input_data["in_cap"]
    network_resolution = input_data["network_resolution"]

    ex_grid = create_ex_grid(ex_grid_data_json)

    n_supply_dict = {
        v["id"]: {"coords": tuple(v["coords"]), "cap": v["cap"]} for v in n_supply_list
    }

    n_demand_dict = {
        v["id"]: {"coords": tuple(v["coords"]), "cap": v["cap"]} for v in n_demand_list
    }

    coords_list = []
    for v in n_supply_list:
        for v1 in n_demand_list:
            coords_list.append(Point([v["coords"][1], v["coords"][0]]))
            coords_list.append(Point([v1["coords"][1], v1["coords"][0]]))

    return (
        n_supply_dict,
        n_demand_dict,
        ex_grid,
        in_cap,
        network_resolution,
        coords_list,
    )


## Prepare Output Data to Wrapper
def prepare_output(nodes, edges, road_nw, n_demand_dict, n_supply_dict):

    clean_nodes = remove_nonjson(nodes)
    clean_edges = remove_nonjson(edges)
    road_nw_json = jsonpickle.encode(road_nw)

    return {
        "nodes": clean_nodes,
        "edges": clean_edges,
        "road_nw": road_nw,
        "n_demand_dict": n_demand_dict,
        "n_supply_dict": n_supply_dict,
    }


## Utilities
def remove_nonjson(output_data):

    if isinstance(output_data, list):
        for datum in output_data:
            to_del = []
            for i in datum.keys():
                if hasattr(datum[i], "__dict__"):
                    to_del.append(i)

            for i in to_del:
                del datum[i]
    else:
        to_del = []
        for i in output_data.keys():
            if hasattr(datum[i], "__dict__"):
                to_del.append(i)

        for i in to_del:
            del datum[i]

    return output_data
