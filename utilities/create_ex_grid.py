#######About##########################
# Author: Bernhard Felber, TU WIEN, Energy Economics Group
# Date: 14.07.21
# Project: Emb3rs

import osmnx as ox
import pandas as pd
import networkx as nx


def create_ex_grid(ex_grid_data):
    ##########CREATE GRAPH OBJECT########

    ex_grid_data = pd.DataFrame.from_dict(ex_grid_data)
    ex_grid = nx.MultiGraph()

    ######GET SINGLE NODE INFORMATION#######

    nodes_1 = ex_grid_data[["from", "lon_from", "lat_from"]]
    nodes_2 = ex_grid_data[["to", "lon_to", "lat_to"]]
    nodes_2.columns = nodes_1.columns

    nodes = pd.concat([nodes_1, nodes_2], axis=0)
    nodes = nodes.drop_duplicates().reset_index()

    ####ADD NODE INFORMATION to GRAPH#######

    for i in nodes.index:
        ex_grid.add_node(
            nodes.iloc[i, 1],
            y=nodes.iloc[i, 3],
            x=nodes.iloc[i, 2],
            osmid=nodes.iloc[i, 1],
        )

    ####ADD EDGE INFORMATION to GRAPH#######

    for i in ex_grid_data.index:
        ex_grid.add_edge(
            ex_grid_data.iloc[i, 0],
            ex_grid_data.iloc[i, 3],
            length=ex_grid_data.iloc[i, -2],
            surface_pipe=ex_grid_data.iloc[i, -1],
            inner_diameter_existing_grid_element=ex_grid_data.iloc[i, -4],
            costs_existing_grid_element=ex_grid_data.iloc[i, -3],
        )

    ####SET GENERAL GRAPH INFORMATION#######

    ex_grid.graph["crs"] = "epsg:4326"
    ex_grid.graph["name"] = "ex_grid"

    ####CREATE GDFS FROM NODES AND EDGES TO CREATE A GEOMETRY ATTRIBUTE#######

    nodes, edges = ox.graph_to_gdfs(ex_grid)
    ex_grid = ox.graph_from_gdfs(nodes, edges)

    return ex_grid
