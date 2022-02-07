#######About##########################
#Author: Bernhard Felber, TU WIEN, Energy Economics Group
#Date: 16.07.21
#Project: Emb3rs

################################################
################Packages to load################
################################################

import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from pyomo.environ import *
from pyomo.opt import *
import numpy as np
import folium
import haversine as hs
from shapely.geometry import Polygon, LineString, Point
import sklearn
import matplotlib.pyplot as plt
import os


##########Function for creating a network solution. It requires a polygon for defining the OSM area to load, a dictionary of supply point (source)
##########and a dictionary of demand points (sinks) as mandatory inputs. These are defined in the code above but have to come from the core functionalites module
##########The existing grid object is optional, the exchange capacities are optional and the network_resolution is also optional

class GIS:

    def create_network(coords_list, N_supply_dict, N_demand_dict, ex_grid = nx.MultiDiGraph(), Ex_cap = pd.DataFrame(), network_resolution= "high"):

        polygon = Polygon(coords_list).convex_hull
        
        if len(Ex_cap) == 0:
            pass
        else:
            Ex_cap.iloc[:, 3:len(Ex_cap.columns)] = Ex_cap.iloc[:, 3:len(Ex_cap.columns)] / 1000
            ################################################################################
            ######ITERATION TEO - DELETE SOURCES/SINKS THAT WERE EXCLUDED FROM TEO##########

            Source_list_TEO = Ex_cap.copy()[Ex_cap["Classification type"] == "Source"]
            Source_list_TEO["sum"] = Source_list_TEO.iloc[:, 3:-1].sum(axis=1)  #####hier weitermachen
            Source_list_TEO = Source_list_TEO[Source_list_TEO["sum"] > 0].drop("sum", axis=1)
            Source_list_TEO = list(Source_list_TEO["Number"])

            Sources_to_delete = set(list(N_supply_dict)) - set(list(Source_list_TEO))

            if len(Source_list_TEO) > 0:
                for i in Sources_to_delete:
                    del N_supply_dict[i]

            Sink_list_TEO = Ex_cap.copy()[Ex_cap["Classification type"] == "Sink"]
            Sink_list_TEO["sum"] = Sink_list_TEO.iloc[:, 3:-1].sum(axis=1)
            Sink_list_TEO = Sink_list_TEO[Sink_list_TEO["sum"] > 0]
            Sink_list_TEO = list(Sink_list_TEO["Number"])

            Sinks_to_delete = set(list(N_demand_dict)) - set(list(Sink_list_TEO))

            if len(Sink_list_TEO) > 0:
                for i in Sinks_to_delete:
                    del N_demand_dict[i]


        ################################################################################
        ######GET OSM ROAD NETWORK######################################################

        if network_resolution == "high":

            cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential|living_street|service|pedestrian|unclassified|track|road|path"]'

            road_nw = ox.graph_from_polygon(polygon, simplify=False, clean_periphery=True, custom_filter=cf)
            road_nw = ox.simplify_graph(road_nw)

        elif network_resolution == "low":

            cf = '["highway"~"primary|primary_link|secondary|secondary_link"]'

            road_nw = ox.graph_from_polygon(polygon, simplify=False, clean_periphery=True, custom_filter=cf)
            road_nw = ox.simplify_graph(road_nw)
        else:
            pass

        ###########REMOVE LONGER EDGES BETWEEN POINTS IF MUTIPLE EXIST#######################
        ##AS ONLY ONE (u,v - v,u) EDGE BETWEEN TWO POINTS CAN BE CONSIDERED FOR OPTIMIZATION#

        nodes, edges = ox.graph_to_gdfs(road_nw)

        network_edges_crs = edges.crs
        network_nodes_crs = nodes.crs

        edges_to_drop = []
        edges_double = pd.DataFrame(edges)
        edges_double["id"] = edges_double["u"].astype(str) + "-" + edges_double["v"].astype(str)

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

        edges_one_way = pd.DataFrame(edges[edges["oneway"] == True])
        edges_one_way["id"] = list(zip(edges_one_way["u"], edges_one_way["v"]))

        edges_to_drop = []

        for i in edges_one_way["id"]:
            edges_u_v = edges_one_way[(edges_one_way["u"] == i[0]) & (edges_one_way["v"] == i[1])]
            edges_v_u = edges_one_way[(edges_one_way["u"] == i[1]) & (edges_one_way["v"] == i[0])]
            edges_all = pd.concat([edges_u_v, edges_v_u])
            if len(edges_all) > 1:
                mx_ind = edges_all["length"].idxmin()
                mx = edges_all.drop(mx_ind)
                edges_to_drop.append(mx)
            else:
                None

        try:
            edges_to_drop = pd.concat(edges_to_drop).drop("id", axis=1)
            edges_to_drop = edges_to_drop[~edges_to_drop.index.duplicated(keep='first')]
            edges_to_drop = edges_to_drop.drop_duplicates(subset=['length'], keep='last')
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
            ex_grid = ox.gdfs_to_graph(nodes_ex_grid, edges_ex_grid)

            ################################################################################
            ######FIND CLOSEST POINTS BETWEEN OSM GRAPH AND EX GRID#########################

            dist = sklearn.neighbors.DistanceMetric.get_metric("haversine")
            nodes_ex = ox.graph_to_gdfs(ex_grid, nodes=True, edges=False)
            nodes_rn = ox.graph_to_gdfs(road_nw, nodes=True, edges=False)

            dist_matrix = dist.pairwise(nodes_ex[["y", "x"]], nodes_rn[["y", "x"]])
            dist_matrix = pd.DataFrame(dist_matrix, index=nodes_ex.index, columns=nodes_rn.index)

            ################################################################################
            ######RETRIEVE INDEX OF MIN VALUE FROM DISTANCE MATRIX##########################

            col = dist_matrix.min().idxmin()
            ind = dist_matrix.loc[:, col].idxmin()


            ################################################################################
            ####ADD NECESSARY ATTRIBUTES TO EXISTING GRAPH OBJECT TO PERFORM CALCULATION####

            nx.set_edge_attributes(ex_grid, "None", "surface_type")
            nx.set_edge_attributes(ex_grid, 0, "restriction")
            #nx.set_edge_attributes(ex_grid, 0,"surface_pipe")  ###maybe this could possibly be already part of upload if we allow surface pipes
            nx.set_edge_attributes(ex_grid, 1, "existing_grid_element")
            #nx.set_edge_attributes(ex_grid, 0.5,"diameter_existing_grid_element")  ###this attribute already needs to come from the upload, maybe the name needs to be transformed
            #nx.set_edge_attributes(ex_grid, 0, "costs_existing_grid_element")  ###this attribute already needs to come from the upload, maybe the name needs to be transformed

            ################################################################################
            ######COMBINE BOTH GRAPHS AND ADD EDGE WITH DISTANCE ATTRIBUTE##################

            road_nw = nx.compose(ex_grid, road_nw)
            road_nw.add_edge(ind, col, length=hs.haversine((road_nw.nodes[ind]["y"], road_nw.nodes[ind]["x"]),
                                                           (road_nw.nodes[col]["y"], road_nw.nodes[col]["x"])) * 1000,
                             surface_type="street",
                             restriction=0, surface_pipe=0, existing_grid_element=0,
                             inner_diameter_existing_grid_element=0, costs_existing_grid_element=0)

        road_nw_streets = road_nw.copy()

        ################################################################################
        ######CONNECT SOURCES AND SINKS TO OSM GRAPH####################################

        for k, v in N_supply_dict.items():
            dist_edge = ox.get_nearest_edge(road_nw_streets, (v["coords"][0], v["coords"][1]))
            dist_1 = ox.euclidean_dist_vec(v["coords"][0], v["coords"][1], road_nw_streets.nodes[dist_edge[1]]["y"],
                                           road_nw_streets.nodes[dist_edge[1]]["x"])
            dist_2 = ox.euclidean_dist_vec(v["coords"][0], v["coords"][1], road_nw_streets.nodes[dist_edge[2]]["y"],
                                           road_nw_streets.nodes[dist_edge[2]]["x"])
            dist_dict = {road_nw_streets.nodes[dist_edge[1]]["osmid"]: dist_1,
                         road_nw_streets.nodes[dist_edge[2]]["osmid"]: dist_2}
            point_to_connect = min(dist_dict, key=dist_dict.get)
            road_nw.add_node(k, y=v["coords"][0], x=v["coords"][1], osmid=k)
            road_nw.add_edge(k, point_to_connect, length=hs.haversine((v["coords"][0], v["coords"][1]), (
            road_nw.nodes[point_to_connect]["y"], road_nw.nodes[point_to_connect]["x"])) * 1000, surface_type="street",
                             restriction=0, surface_pipe=0, existing_grid_element=0,
                             inner_diameter_existing_grid_element=0, costs_existing_grid_element=0)

        for k, v in N_demand_dict.items():
            dist_edge = ox.get_nearest_edge(road_nw_streets, (v["coords"][0], v["coords"][1]))
            dist_1 = ox.euclidean_dist_vec(v["coords"][0], v["coords"][1], road_nw_streets.nodes[dist_edge[1]]["y"],
                                           road_nw_streets.nodes[dist_edge[1]]["x"])
            dist_2 = ox.euclidean_dist_vec(v["coords"][0], v["coords"][1], road_nw_streets.nodes[dist_edge[2]]["y"],
                                           road_nw_streets.nodes[dist_edge[2]]["x"])
            dist_dict = {road_nw_streets.nodes[dist_edge[1]]["osmid"]: dist_1,
                         road_nw_streets.nodes[dist_edge[2]]["osmid"]: dist_2}
            point_to_connect = min(dist_dict, key=dist_dict.get)
            road_nw.add_node(k, y=v["coords"][0], x=v["coords"][1], osmid=k)
            road_nw.add_edge(k, point_to_connect, length=hs.haversine((v["coords"][0], v["coords"][1]), (
            road_nw.nodes[point_to_connect]["y"], road_nw.nodes[point_to_connect]["x"])) * 1000,surface_type="street",
                             restriction=0, surface_pipe=0, existing_grid_element=0,
                             inner_diameter_existing_grid_element=0, costs_existing_grid_element=0)


        ################################################################################
        ####PROJECT GRAPH AND TURN INTO UNDIRECTED FOR USER DISPLAY#####################

        road_nw = ox.get_undirected(road_nw)
        road_nw = ox.project_graph(road_nw)

        ################################################################################
        ####################DELETE ATTRIBUTES FROM EDGES NOT NEEDED#####################

        nodes, edges = ox.graph_to_gdfs(road_nw)
        cols_to_drop = ['osmid', 'oneway', 'name', 'highway', 'maxspeed', 'lanes', 'junction', 'service', 'access']
        edges = edges.drop(edges.columns.intersection(cols_to_drop), axis=1)
        road_nw = ox.gdfs_to_graph(nodes, edges)

        return road_nw


    def optimize_network(road_nw, N_supply_dict, N_demand_dict, water_den, factor_street_terrain, factor_street_overland, heat_capacity, flow_temp, return_temp, surface_losses_df, ground_temp, ambient_temp,
                         fc_dig_st, vc_dig_st, vc_dig_st_ex, fc_dig_tr, vc_dig_tr, vc_dig_tr_ex, fc_pip, vc_pip, vc_pip_ex, invest_pumps, Ex_cap = pd.DataFrame()):

        road_nw_area = road_nw.copy()
        nodes, edges = ox.graph_to_gdfs(road_nw)

        ################################################################################
        ###########CONVERT GRAPH TO NX GRAPH FOR FURTHER PROCESSING#####################

        road_nw = nx.Graph(road_nw)  ###attention - if there would still be two edges in one direction connecting two point, the function would remove one of them
        #nx.draw(road_nw)

        ################################################################################
        ###########ADD EDGE ATTRIBUTES THAT SHOULD BE PART OF GRAPH ALREADY#############

        ################################################################################
        # CONVERT GRAPH TO DICT FOR PYOMO INPUT AND ONLY RETRIEVE DATA NEEDED ###########

        road_nw_data = nx.to_dict_of_dicts(road_nw)

        road_nw_data_points = [(x, y) for x in road_nw_data for y in road_nw_data[x]]

        road_nw_data_edges = []
        for k, v in road_nw_data.items():
            for k1, v1 in v.items():
                road_nw_data_edges.append([v1["length"], v1["surface_type"], v1["restriction"], v1["surface_pipe"],
                                           v1["existing_grid_element"], v1["inner_diameter_existing_grid_element"],
                                           v1["costs_existing_grid_element"]])
                road_nw_data_names = list(v1.keys())

        road_nw_data = dict(zip(road_nw_data_points, road_nw_data_edges))

        #road_nw_data[(25877743, 25877763)][5] = 0.02

        ################################################################################
        #########INVERT ALL KEYS TO BE SURE EACH EDGE EXISTS IN BOTH DIRECTIONS#########

        for (i, j) in road_nw_data.keys():
            road_nw_data[(j, i)] = road_nw_data[(i, j)]

        ################################################################################
        #########DEFINITION OF SOURCES AND SINKS FOR PYOMO##############################

        N = list(nodes.index)
        N_all_so_si = list(N_supply_dict.keys()) + list(N_demand_dict.keys())
        N_supply = list(N_supply_dict.keys())
        N_demand = list(N_demand_dict.keys())

        ################################################################################
        ######################PREPARE NODE PARAMETERS FOR PYOMO#########################

        ###########TURN DICTS INTO DFS##################################################

        supply_data_py = pd.DataFrame.from_dict(N_supply_dict, orient="index")
        demand_data_py = pd.DataFrame.from_dict(N_demand_dict, orient="index")

        ###########DECIDE FOR THE CORRECT EXCHANGE CAPACITIES IF THE SUM OF CAPACITY SINKS
        ###### DOES NOT MATCH SUM OF CAPACITIES SOURCES (APPLIES FOR CASES THE GIS HAS NOT
        ################## ITERATED WITH THE TEO YET#################

        new_cap = []
        if supply_data_py.cap.sum() > demand_data_py.cap.sum():
            for i in supply_data_py.cap:
                z = i
                diff = (supply_data_py.cap.sum() - demand_data_py.cap.sum())
                share = z / supply_data_py.cap.sum()
                z = z - (diff * share)
                new_cap.append(z)
        elif supply_data_py.cap.sum() < demand_data_py.cap.sum():
            for i in demand_data_py.cap:
                z = i
                diff = (demand_data_py.cap.sum() - supply_data_py.cap.sum())
                share = z / demand_data_py.cap.sum()
                z = z - (diff * share)
                new_cap.append(z)
        else:
            print("Supply capacity == demand capacity")

        ##################WRITE CORRECT VALUES TO DFS#################

        if supply_data_py.cap.sum() > demand_data_py.cap.sum():
            supply_data_py.cap = new_cap
        elif supply_data_py.cap.sum() < demand_data_py.cap.sum():
            demand_data_py.cap = new_cap
        else:
            print("No adapted exchange capacities")

        ##################CONCAT DFS WITH ADAPTED EX CAPACITIES########

        data_py = pd.concat([supply_data_py, demand_data_py], axis=0)
        list_of_nodes = pd.DataFrame(N)

        ###########CREATE DF WITH THE CORRECT EX. CAPACITY INFORMATION#######

        supply_data_py = list_of_nodes.merge(supply_data_py["cap"], left_on=list_of_nodes[0],
                                             right_on=supply_data_py.index, how="left").drop("key_0", axis=1)
        demand_data_py = list_of_nodes.merge(demand_data_py["cap"], left_on=list_of_nodes[0],
                                             right_on=demand_data_py.index, how="left").drop("key_0", axis=1)
        data_py = pd.concat([supply_data_py, demand_data_py["cap"]], axis=1)
        data_py.columns = ["Nodes", "cap_sup", "cap_dem"]

        ###########ADD INPUTS FOR CONNECTIVITY RULE##########################

        list_con_dem = []
        for n in N:
            if n in N_all_so_si[1:len(N_all_so_si)]:
                list_con_dem.append(1)
            else:
                list_con_dem.append(0)

        data_py["con_dem"] = list_con_dem

        list_con_sup = []
        for n in N:
            if n in [N_all_so_si[0]]:
                list_con_sup.append(len(N_all_so_si[1:len(N_all_so_si)]))
            else:
                list_con_sup.append(0)

        data_py["con_sup"] = list_con_sup
        data_py = data_py.fillna(0)
        data_py.index = data_py.Nodes
        del (supply_data_py, demand_data_py)

        ################################################################################
        #################GET COPY OF ROAD_NW_DATA TO PRESERVE CORRECT LENGTH###########

        road_nw_data_copy = road_nw_data.copy()

        ################################################################################
        ######################PREPARE EDGE PARAMETERS FOR PYOMO + WEIGTH LENGTH#########

        road_nw_data = pd.DataFrame.from_dict(road_nw_data, orient="index")
        road_nw_data.loc[road_nw_data[1] == "terrain", 0] = road_nw_data[0] * (1 - factor_street_terrain)
        road_nw_data.loc[road_nw_data[1] == "None", 0] = road_nw_data[0] * (1 - factor_street_overland)
        # road_nw_data = road_nw_data.sort_values(by=[1])
        road_nw_data = road_nw_data.to_dict("index")

        ################################################
        ################Pyomo model routing#############
        ################################################

        opt = solvers.SolverFactory("gurobi")
        model = ConcreteModel()

        ###################################################################
        ######################SETS#########################################

        model.node_set = Set(initialize=N)
        model.edge_set = Set(initialize=road_nw_data.keys())
        model.flow_var_set = Set(initialize=np.arange(0, len(N_supply + N_demand)+1, 1))

        ###################################################################
        ######################PARAMETERS EDGES#############################

        model.edge_length = Param(model.edge_set, initialize={key: value[0] for (key, value) in road_nw_data.items()})
        model.edge_restriction = Param(model.edge_set,
                                       initialize={key: value[2] for (key, value) in road_nw_data.items()})
        model.edge_existing_grid = Param(model.edge_set,
                                         initialize={key: value[4] for (key, value) in road_nw_data.items()})

        ###################################################################
        ######################PARAMETERS NODES#############################

        model.node_demand = Param(model.node_set, initialize=data_py["con_dem"].to_dict())
        model.node_supply = Param(model.node_set, initialize=data_py["con_sup"].to_dict())

        ###################################################################
        ######################VARIABLES####################################

        model.flow = Var(model.edge_set, within=model.flow_var_set)
        model.bool = Var(model.edge_set, domain=Binary)

        ###########RECODE FLOW VAR INTO BOOL VAR###########################

        Domain_points = [0., 0., 0.001, 1000.]
        Range_points = [0., 0., 1., 1.]

        model.con = Piecewise(model.edge_set, model.bool, model.flow,
                              pw_pts=Domain_points,
                              pw_constr_type="EQ",
                              f_rule=Range_points,
                              pw_repn="INC")

        ###################################################################
        ######################CONSTRAINT###################################

        ###########CONNECTIVITY CONSTRAINT#################################

        def flow_rule(model, n):
            InFlow = sum(model.flow[i, j] for (i, j) in model.edge_set if j == n)
            OutFlow = sum(model.flow[i, j] for (i, j) in model.edge_set if i == n)

            input = InFlow + model.node_supply[n]
            output = OutFlow + model.node_demand[n]
            return input == output

        model.flow_constraint = Constraint(N, rule=flow_rule)

        ###################################################################
        ######################OBJECTIVE####################################

        ###########TARGET FUNCTION#########################################

        model.result = Objective(expr=sum(model.bool[a] * model.edge_length[a] * 2 for a in model.edge_set),
                                 sense=minimize)

        ###########CONSIDER EDGE RESTRICTION IN RESULT#####################

        for i in model.edge_set:
            if model.edge_restriction[i] == -1:
                model.flow[i].setub(0)
            else:
                model.flow[i].setlb(0)

        for i in model.edge_set:
            if model.edge_restriction[i] == 1:
                model.flow[i].setlb(1)
            else:
                model.flow[i].setlb(0)

        ###########SET EXISTING GRIDS AS PART OF SOLUTION#################

        for i in model.edge_set:
            if model.edge_existing_grid[i] == 1:
                model.flow[i].setlb(1)
            else:
                model.flow[i]

        ###########SOLVE MODEL############################################
        #opt.options[
            #'timelimit'] = 60 * 12  ###max solver solution time, if exceeded the solver stops and takes the best found solution at that point
        results = opt.solve(model, tee=True)
        #model.result.expr()

        ###########GET RESULTS############################################

        result_data = model.bool.get_values()

        result_graph = {k: v for k, v in result_data.items() if v > 0.1}

        for (i, j) in list(result_graph):
            result_graph[(j, i)] = result_graph[(i, j)]

        ############GET THE PIPE DIAMETER FOR EVERY EDGE OF EXISTING GRID####
        ######FILTER ALL ELEMENTS FROM ROAD NETWORK THAT ARE IN SOLUTION OF##
        ##################### RESULT GRAPH###################################


        road_nw_ex_grid = road_nw_data_copy.copy()
        diff_edges_result_graph = set(road_nw_ex_grid) - set(result_graph)
        for diff_edges_result_graph in diff_edges_result_graph: del road_nw_ex_grid[diff_edges_result_graph]

        ####Transform dict of solution edges to df#####

        road_nw_ex_grid = pd.DataFrame.from_dict(road_nw_ex_grid, orient="index")

        ####in order to set a capacity limit on an existing grid edge we need to translate the
        ###pipe diameter into a thermal capacity before. We create a dataframe for conversion####

        ###################################################################
        #######SET UP LOOK UP DF FOR POWER/DIAMETER CONVERSION#############

        MW_dia_con = pd.DataFrame(columns=["Diameter", "v", "A", "MW"])
        MW_dia_con["Diameter"] = np.arange(0.01, 1.001, 0.001)
        MW_dia_con["A"] = ((MW_dia_con["Diameter"] / 2) ** 2) * 3.14159265359
        MW_dia_con["v"] = 4.7617 * (MW_dia_con["Diameter"] / 2) ** 0.3701 - 0.4834
        MW_dia_con["MW"] = (MW_dia_con.A * MW_dia_con.v * water_den * heat_capacity * (
            abs(flow_temp - return_temp))) / 1000
        MW_dia_con["MW"] = round(MW_dia_con["MW"], 2)

        ###################################################################
        #######FIND CORRESPONDING POWER VALUE FOR DIAMETER#################

        MW_list = []
        for i in road_nw_ex_grid[5]:
            index_dia = MW_dia_con["Diameter"].sub(i).abs().idxmin()
            MW_list.append(MW_dia_con["MW"][index_dia])

        road_nw_ex_grid["MW"] = MW_list
        road_nw_ex_grid["MW"] = road_nw_ex_grid["MW"].replace(0, 99999)

        ################################################
        ###Pyomo model flows V1 WITHOUT TEO#############
        ################################################

        if len(Ex_cap) == 0:

            ###########DATA PREPARATION############################################

            data_py = data_py[data_py["Nodes"].isin(list({k[0] for k, v in result_graph.items()}))]
            N = list(data_py.index)

            opt = solvers.SolverFactory("gurobi")
            model_nw = ConcreteModel()

            ###################################################################
            ######################SETS#########################################

            model_nw.node_set = Set(initialize=list({k[0] for k, v in result_graph.items()}))
            model_nw.edge_set = Set(initialize=result_graph.keys())

            ###################################################################
            ######################VARS#########################################

            model_nw.flow = Var(model_nw.edge_set, bounds=(0, 500)) ###max set to thermal capacity of 500 MW
            model_nw.cap_add = Var(model_nw.edge_set, bounds=(0, 500)) ###additional capacity required if bottleneck

            ###################################################################
            ######################PARAMETERS###################################

            model_nw.node_demand = Param(model_nw.node_set, initialize=data_py["cap_dem"].to_dict())
            model_nw.node_supply = Param(model_nw.node_set, initialize=data_py["cap_sup"].to_dict())
            model_nw.edge_capacities = Param(model_nw.edge_set, initialize=road_nw_ex_grid["MW"].to_dict())
            model_nw.edge_length = Param(model_nw.edge_set, initialize=road_nw_ex_grid[0].to_dict())

            ###################################################################
            ######################CONSTRAINTS##################################

            def flow_rule_nw(model_nw, n):
                InFlow = sum(model_nw.flow[i, j] + model_nw.cap_add[i, j] for (i, j) in model_nw.edge_set if j == n)
                OutFlow = sum(model_nw.flow[i, j] + model_nw.cap_add[i, j] for (i, j) in model_nw.edge_set if i == n)

                input = InFlow + model_nw.node_supply[n]
                output = OutFlow + model_nw.node_demand[n]
                return input == output

            model_nw.flow_constraint = Constraint(N, rule=flow_rule_nw)

            def add_cap_rule(model_nw, i, j):
                return model_nw.flow[i, j] <= model_nw.edge_capacities[i, j] + model_nw.cap_add[i, j]

            model_nw.cap_constraint = Constraint(model_nw.edge_set, rule=add_cap_rule)


            ###################################################################
            ######################OBJECTIVE####################################

            model_nw.result_nw = Objective(expr=sum(
                model_nw.flow[a] * model_nw.edge_length[a]*2 + model_nw.cap_add[a] * 1000000000 for a in
                model_nw.edge_set), sense=minimize)
            result_nw = opt.solve(model_nw, tee=True)

            ###################################################################
            ######################GET RESULTS##################################

            result_data_flow = model_nw.flow.get_values()
            result_data_cap_add = model_nw.cap_add.get_values()
            keys_list = list(result_data_flow.keys())

            ###################################################################
            ######SAVE RESULTS FROM FLOW AND ADD CAP INTO ONE OBJECT###########

            result_data = {}

            for i in keys_list:
                result_data[i] = result_data_flow[i] + result_data_cap_add[i]


        ################################################
        ###Pyomo model flows WITH TEO###################
        ################################################
        else:

            ###########DATA PREPARATION############################################
            #######FIGURE OUT EX CAPACITIES SEPERATED BY SOURCES AND SINKS#########

            Ex_cap_sinks = Ex_cap[Ex_cap["Classification type"] == "Sink"]
            Ex_cap_sinks = Ex_cap_sinks.iloc[:, 2:len(Ex_cap.columns)]
            Ex_cap_sinks.index = Ex_cap_sinks["Number"]
            Ex_cap_sinks = Ex_cap_sinks.drop("Number", axis=1)

            Ex_cap_sources = Ex_cap[Ex_cap["Classification type"] == "Source"]
            Ex_cap_sources = Ex_cap_sources.iloc[:, 2:len(Ex_cap.columns)]
            Ex_cap_sources.index = Ex_cap_sources["Number"]
            Ex_cap_sources = Ex_cap_sources.drop("Number", axis=1)

            ########GET ALL TIME STEPS FROM THE TEO ###############
            TS = Ex_cap.iloc[:, 3:len(Ex_cap.columns)].columns
            # CREATE LIST WHERE ALL RESULTS FROM ITERATIONS ARE SAVED ##

            result_data_all_TS = []

            ########LOOP THROUGH EACH TIME STEP AND ###############
            #############PUT PARAMS TOGETHER IN DF#################

            for i in TS:
                data_py = data_py[data_py["Nodes"].isin(list({k[0] for k, v in result_graph.items()}))]
                data_py = pd.DataFrame(data_py["Nodes"])

                N = list(data_py.index)

                data_py = data_py.merge(Ex_cap_sources[i], left_on=data_py["Nodes"], right_on=Ex_cap_sources.index,
                                        how="left").drop("key_0", axis=1)
                data_py = data_py.merge(Ex_cap_sinks[i], left_on=data_py["Nodes"], right_on=Ex_cap_sinks.index,
                                        how="left").drop("key_0", axis=1)
                data_py.index = data_py["Nodes"]
                data_py = data_py.fillna(0)

                #############SET UP MODEL###########################
                opt = solvers.SolverFactory("gurobi")
                model_nw = ConcreteModel()

                ###################################################################
                ######################SETS#########################################

                model_nw.node_set = Set(initialize=list({k[0] for k, v in result_graph.items()}))
                model_nw.edge_set = Set(initialize=result_graph.keys())

                ###################################################################
                ######################VARS#########################################

                model_nw.flow = Var(model_nw.edge_set, bounds=(0, 500)) ###max set to thermal capacity of 500 MW
                model_nw.cap_add = Var(model_nw.edge_set,bounds=(0, 500))  ###additional capacity required if bottleneck

                ###################################################################
                ######################PARAMETERS###################################

                model_nw.node_demand = Param(model_nw.node_set, initialize=data_py.iloc[:, 2].to_dict())
                model_nw.node_supply = Param(model_nw.node_set, initialize=data_py.iloc[:, 1].to_dict())
                model_nw.edge_capacities = Param(model_nw.edge_set, initialize=road_nw_ex_grid["MW"].to_dict())
                model_nw.edge_length = Param(model_nw.edge_set, initialize=road_nw_ex_grid[0].to_dict())

                ###################################################################
                ######################CONSTRAINTS##################################

                def flow_rule_nw(model_nw, n):
                    InFlow = sum(model_nw.flow[i, j] + model_nw.cap_add[i, j] for (i, j) in model_nw.edge_set if j == n)
                    OutFlow = sum(model_nw.flow[i, j] + model_nw.cap_add[i, j] for (i, j) in model_nw.edge_set if i == n)

                    input = InFlow + model_nw.node_supply[n]
                    output = OutFlow + model_nw.node_demand[n]
                    return input == output

                model_nw.flow_constraint = Constraint(N, rule=flow_rule_nw)

                def add_cap_rule(model_nw, i, j):
                    return model_nw.flow[i, j] <= model_nw.edge_capacities[i, j] + model_nw.cap_add[i, j]

                model_nw.cap_constraint = Constraint(model_nw.edge_set, rule=add_cap_rule)

                ###################################################################
                ######################OBJECTIVE####################################

                model_nw.result_nw = Objective(expr=sum(
                    model_nw.flow[a] * model_nw.edge_length[a]*2 + model_nw.cap_add[a] * 1000000000 for a in
                    model_nw.edge_set), sense=minimize)
                result_nw = opt.solve(model_nw, tee=True)

                ###################################################################
                ######################GET RESULTS##################################

                result_data_flow = model_nw.flow.get_values()
                result_data_cap_add = model_nw.cap_add.get_values()
                keys_list = list(result_data_flow.keys())

                ###################################################################
                ######SAVE RESULTS FROM FLOW AND ADD CAP INTO ONE OBJECT###########

                result_data = {}

                for i in keys_list:
                    result_data[i] = result_data_flow[i] + result_data_cap_add[i]

                result_data_all_TS.append(pd.DataFrame.from_dict(result_data, orient="index"))

                del (model_nw)

            result_data = pd.concat(result_data_all_TS, axis=1)
            result_data["max"] = result_data.max(axis=1)
            result_data["max"] = round(result_data["max"], 2)
            result_data = result_data["max"].to_dict()

        ###################################################################
        ##################CHECK IF EVERY EDGE HAS A FLOW###################

        ###########GET EDGES THAT HAVE A FLOW > 0 ON THEM##################

        graph_test_data = result_data.copy()
        graph_test_data = {k: v for k, v in graph_test_data.items() if v != 0}

        ###########CREATE GRAPH OUT OF ONLY EDGES WITH FLOW ON IT##########

        graph_test = nx.Graph()
        graph_test.add_edges_from(graph_test_data.keys())
        nx.draw(graph_test)

        ###########GET NUMBER OF SUBGRAPHS IN GRAPH########################

        subgraphs = len(list((graph_test.subgraph(c) for c in nx.connected_components(graph_test))))

        if subgraphs > 1:
            ############################################################################
            #######ASSIGN VALUES TO EDGES WITH NO FLOW IF MULTIPLE SUBGRAPHS EXIST######
            ############################################################################

            ###################################################################
            #############FIND EDGES WITH NO FLOW ON IT#########################

            for (i, j) in result_data.keys():
                if (result_data[(i, j)] + result_data[(j, i)]) == 0:
                    result_data[(i, j)] = 99999
                else:
                    result_data[(i, j)] = result_data[(i, j)]

            zero_values = {k: v for k, v in result_data.items() if v == 99999}
            zero_values = pd.DataFrame.from_dict(zero_values.keys())
            zero_values = zero_values.stack()
            zero_values = zero_values.value_counts()  ###maybe cases occur where there is exactly the same start and end point
            zero_values = zero_values[zero_values == 1]
            conn_points = zero_values.index
            del (zero_values)

            graph_data = {k: v for k, v in result_data.items() if v > 0}
            graph = nx.Graph()
            for k, v in graph_data.items():
                graph.add_edge(k[0], k[1], weight=v)

            adj_list = []
            for i in conn_points:
                adj_df = pd.DataFrame.from_dict(graph[i].values())
                adj_df.columns.values[0] = i
                adj_list.append(adj_df)

            adj_df = pd.concat(adj_list, axis=1)
            adj_df = adj_df.replace(to_replace=99999, value=np.NaN)
            adj_df = adj_df.mean()
            adj_df_help = adj_df.copy()

            graph_data = {k: v for k, v in result_data.items() if v == 99999}
            graph = nx.Graph()
            for k, v in graph_data.items():
                graph.add_edge(k[0], k[1], weight=v)

            con_list = []
            for i in range(0, len(adj_df)):
                for z in range(0, len(adj_df_help)):
                    con = nx.has_path(graph, adj_df.index[i], adj_df_help.index[z])
                    con_list.append(
                        [adj_df.index[i], adj_df.values[i], adj_df_help.index[z], adj_df_help.values[z], con])

            con_df = pd.DataFrame(con_list)
            con_df = con_df[con_df[0] != con_df[2]]
            con_df = con_df[con_df[4] == True].reset_index()
            con_df["sum_nodes"] = con_df[0].astype(int) + con_df[2].astype(int)
            index_to_keep = con_df["sum_nodes"].drop_duplicates().index
            con_df["ex_capacity"] = (con_df[1] + con_df[3]) / 2
            con_df = con_df.iloc[index_to_keep, :]

            graph = graph.to_directed()

            for pois in range(0, len(con_df)):
                node = con_df[0].iloc[pois]
                cap = con_df["ex_capacity"].iloc[pois]
                graph.nodes[node]["demand"] = cap

            for pois in range(0, len(con_df)):
                node = con_df[2].iloc[pois]
                cap = con_df["ex_capacity"].iloc[pois]
                graph.nodes[node]["demand"] = cap * -1

            flow_dict = nx.min_cost_flow(graph, demand="demand")

            all_tuples = []
            for k, v in flow_dict.items():
                for k1, v1 in v.items():
                    all_tuples.append([tuple([k, k1]), v1])

            all_edges_dict = dict(zip([n[0] for n in all_tuples], [n[1] for n in all_tuples]))
            all_edges_dict = {k: v for k, v in all_edges_dict.items() if v != 0}
            all_edges_dict.update({k: v for k, v in result_data.items() if v != 0 and v != 99999})
        else:
            result_data = {k: v for k, v in result_data.items() if v != 0}
            all_edges_dict = result_data

        ###################################################################
        #####GET RESULTS, CONVERT IT TO DF AND MERGE IT WITH ALL EDGE ATTR.
        #######FROM GRAPH##################################################

        result_df = pd.DataFrame.from_dict(all_edges_dict, orient="index", columns=["MW"])
        graph_df = pd.DataFrame.from_dict(road_nw_data_copy, orient="index")
        result_df = result_df.merge(graph_df, left_on=result_df.index, right_on=graph_df.index, how="left")
        result_df.index = result_df["key_0"]
        result_df = result_df.drop(["key_0", 2, 4], axis=1)

        #####Merge information about the capacity limits for existing grid elements######
        result_df = result_df.merge(road_nw_ex_grid["MW"], left_on=result_df.index, right_on=road_nw_ex_grid.index,
                                    how="left")
        result_df.index = result_df["key_0"]
        result_df = result_df.drop(["key_0"], axis=1)

        result_df.columns = ["MW", "Length", "Surface_type", "Surface_pipe", "Diameter_ex_grid", "Costs_ex_grid",
                             "Capacity_limit"]
        result_df.loc[result_df["Capacity_limit"] == 99999, "Capacity_limit"] = "None"
        del (graph_df)

        ###############################################################################
        ########DETERMINE CONNECTING ELEMENTS BETWEEN ALL SOURCES AND SINKS############
        ###############################################################################

        all_edges = list(result_df.index)
        u = [n[0] for n in all_edges]
        v = [n[1] for n in all_edges]
        result_df["u"] = u
        result_df["v"] = v
        graph_solution = nx.from_pandas_edgelist(result_df, source="u", target="v", edge_attr="Length")

        ###########################################################################
        #######FIND SHORTEST PATH BETWEEN ALL SOURCE/SINK PAIRS IN ONE DIRECTION###

        shortest_paths = {}
        for i in N_supply:
            for x in N_demand:
                shortest_paths[i, x] = nx.shortest_path(G=graph_solution, source=i, target=x, weight="Length")

        ###################################################################
        #######CONVERT ALL SOURCE/SINK PAIR DICT TO DF#####################

        shortest_paths = pd.DataFrame.from_dict(shortest_paths, orient="index").transpose()

        ###################################################################
        #######CREATE EDGE STRUCTURE FROM SHORTEST PATH DF#################

        shortest_paths_copy = pd.DataFrame(columns=shortest_paths.columns, index=range(len(shortest_paths) - 1))

        for i in range(0, len(shortest_paths) - 1):
            for z in range(0, len(shortest_paths.columns)):
                if shortest_paths.iloc[i, z] and shortest_paths.iloc[i + 1, z] > 0:
                    shortest_paths_copy.iloc[i, z] = tuple([shortest_paths.iloc[i, z], shortest_paths.iloc[i + 1, z]])
                else:
                    shortest_paths_copy.iloc[i, z] = 0

        shortest_paths_ij = shortest_paths_copy.copy()
        del (shortest_paths_copy)

        ###########################################################################
        #######FIND SHORTEST PATH BETWEEN ALL SOURCE/SINK PAIRS IN OTHER DIRECTION#

        shortest_paths_ji = shortest_paths_ij.copy()

        for i in range(0, len(shortest_paths_ij)):
            for z in range(0, len(shortest_paths_ij.columns)):
                if shortest_paths_ji.iloc[i, z] != 0:
                    shortest_paths_ji.iloc[i, z] = tuple(
                        [shortest_paths_ij.iloc[i, z][1], shortest_paths_ij.iloc[i, z][0]])
                else:
                    shortest_paths_ji.iloc[i, z] = 0

        ###################################################################
        ####COMBINE TWO DFS WITH SHORTEST PATH INFORMATION#################

        ###########ASSIGN SAME COLUMN NAMES TO BOTH DFS TO CONCAT THEM#############

        shortest_paths = pd.concat([shortest_paths_ij, shortest_paths_ji], axis=0).reset_index().drop("index", axis=1)
        shortest_paths.columns = [str(n) for n in list(shortest_paths.columns)]

        ###########MERGE EDGES SOLUTION AND SHORTEST PATH EACH SOURCE/SINK PAIR###

        rows_list = []
        for i in range(0, len(shortest_paths.columns)):
            dict1 = {}
            merge_df = result_df.merge(shortest_paths.iloc[:, i], left_on=result_df.index,
                                       right_on=shortest_paths.iloc[:, i], how="left")
            dict1.update(merge_df)
            rows_list.append(pd.DataFrame(dict1))

        ###########COMBINE INFORMATION INTO SINGLE DF###########################

        result_df = pd.concat(rows_list, axis=1)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        ################################################
        ################Calculate losses################
        ################################################

        ###################################################################
        #######SET UP LOOK UP DF FOR POWER/DIAMETER CONVERSION#############

        MW_dia_con = pd.DataFrame(columns=["Diameter", "v", "A", "MW"])
        MW_dia_con["Diameter"] = np.arange(0.01, 1.001, 0.001)
        MW_dia_con["A"] = ((MW_dia_con["Diameter"] / 2) ** 2) * 3.14159265359
        MW_dia_con["v"] = 4.7617 * (MW_dia_con["Diameter"] / 2) ** 0.3701 - 0.4834
        MW_dia_con["MW"] = (MW_dia_con.A * MW_dia_con.v * water_den * heat_capacity * (
            abs(flow_temp - return_temp))) / 1000
        MW_dia_con["MW"] = round(MW_dia_con["MW"], 3)
        ###################################################################
        #######FIND CORRESPONDING DIAMETER FOR POWER VALUE#################

        diameter_list = []
        for i in result_df["MW"]:
            index_dia = MW_dia_con["MW"].sub(i).abs().idxmin()
            diameter_list.append(MW_dia_con["Diameter"][index_dia])

        result_df["Diameter"] = diameter_list

        ###################################################################
        #######OVERRULE DIAMETER BY EXISTING DIAMETER IF AVAILABLE#########

        for i in range(0, len(result_df)):
            if result_df["Diameter_ex_grid"][i] != 0:
                result_df.loc[i, "Diameter"] = result_df.loc[i,"Diameter_ex_grid"]
            else:
                pass

        result_df["Diameter"] = round(result_df["Diameter"], 3)

        ###################################################################
        #########################LOSS CALCULATION##########################

        loss_list = []
        for i in range(0, len(result_df)):
            if result_df["Surface_pipe"][i] == 1:
                index_dia = surface_losses_df["DN"].sub(result_df["Diameter"][i]).abs().idxmin()
                loss_list.append(surface_losses_df['Overland losses in W/mK'][index_dia] * abs(
                    (((flow_temp + return_temp) / 2) - ambient_temp)))
            else:
                loss_list.append((abs((flow_temp + return_temp) / 2 - ground_temp) * (
                            0.1685 * np.log(result_df["Diameter"][i]) + 0.85684)))

        result_df["Losses [W/m]"] = loss_list
        result_df["Losses [W/m]"] = round(result_df["Losses [W/m]"], 3)
        result_df["Length"] = round(result_df["Length"] * 2, 2)
        result_df["Losses [W]"] = result_df["Losses [W/m]"] * result_df["Length"]
        result_df["Losses [W]"] = round(result_df["Losses [W]"], 3)

        ################################################
        ################Calculate costs#################
        ################################################

        result_df.loc[result_df["Surface_type"] == "street", "costs_digging"] = (fc_dig_st + (
                    vc_dig_st * result_df["Diameter"]) ** vc_dig_st_ex) * result_df["Length"]
        result_df.loc[result_df["Surface_type"] == "terrain", "costs_digging"] = (fc_dig_tr + (
                    vc_dig_tr * result_df["Diameter"]) ** vc_dig_tr_ex) * result_df["Length"]
        result_df.loc[result_df["Surface_type"] == "None", "costs_digging"] = 0
        result_df["costs_piping"] = (fc_pip + (vc_pip * result_df["Diameter"]) ** vc_pip_ex) * result_df["Length"]
        result_df["cost_total"] = round(result_df["costs_piping"] + result_df["costs_digging"], 2)
        result_df.loc[result_df['Diameter_ex_grid'] > 0, "cost_total"] = result_df['Costs_ex_grid']

        ###########EXTRACT SOLUTION BY EACH SOURCE/SINK PAIR####################

        so_sin_cols = [n for n in list(result_df.columns) if "(" in n]
        res_sources_sinks = result_df.copy()
        res_sources_sinks = res_sources_sinks.fillna(0)

        rows_list = []
        for i in so_sin_cols:
            df = res_sources_sinks.loc[:, [i, 'cost_total', 'Losses [W]', 'Length', 'MW']]
            df = df[df[i] != 0]
            df_sum = [i, df['Losses [W]'].sum(), df['MW'].sum(), df['Length'].sum(), df['cost_total'].sum()]
            rows_list.append(df_sum)

        res_sources_sinks = pd.DataFrame(rows_list, columns=["From/to", "Losses total [W]", "Installed capacity [MW]",
                                                             "Length [m]", "Total_costs [EUR]"])
        sums_ = pd.Series(
            ["Sum (no redundancy)", result_df['Losses [W]'].sum(), result_df['MW'].sum(), result_df['Length'].sum(),
             result_df["cost_total"].sum() + invest_pumps])
        res_sources_sinks.loc[len(res_sources_sinks)] = sums_.values

        res_sources_sinks.to_excel("Results_GIS.xlsx", index=False)

        #######VISUAL RESULTS##################

        result_df_ij = result_df.copy()
        result_df_ji = result_df.copy()
        result_df_ji["u"] = result_df_ij["v"]
        result_df_ji["v"] = result_df_ij["u"]
        result_df_ji["key_0"] = "(" + result_df_ji["u"].astype(str) + ", " + result_df_ji["v"].astype(str) + ")"

        result_df_result = pd.concat([result_df_ji, result_df_ij], axis=0)

        ############DATA WILL BE DELETED LATER, ONLY FOR CHECK IF GIS WORKS CORRECTLY#################
        ##############################################################################################
        ##############################################################################################
        ##############################################################################################

        edges["from_to"] = "(" + edges["u"].astype(str) + ", " + edges["v"].astype(
            str) + ")"  ####create new variable in edges where solution can be retrieved from

        edges = edges.merge(
            result_df_result[["MW", "Length", "Surface_type", "cost_total", "Diameter", 'Losses [W/m]', 'Losses [W]',
                              'Capacity_limit']],left_on=edges["from_to"], right_on=result_df_result["key_0"].astype(str), how="left")
        edges["MW"] = edges["MW"].fillna(0)

        edges_solution = edges[edges["MW"] != 0]

        nodes_to_filter = pd.DataFrame()
        nodes_to_filter = pd.concat([edges_solution["u"], edges_solution["v"]],
                                    axis=0)  ###create a vector out of all solution points (source and/or target)
        nodes_to_filter = nodes_to_filter.unique()  ###create a vector with only unique points (no double counts) - all point that are part of solution regardless if source or target points

        nodes_solution = nodes[nodes["osmid"].isin(nodes_to_filter)]  ###filter solution points from input graph

        edges_solution  ###edges of solution network
        nodes_solution  ###nodes of solution network

        #network = ox.gdfs_to_graph(nodes_solution, edges_solution)

        m = folium.Map(
            location=[list(N_demand_dict.values())[0]["coords"][0], list(N_demand_dict.values())[0]["coords"][1]],
            zoom_start=11, control_scale=True)  #####create basemap
        #####layer for whole routing area####
        style = {'fillColor': '#00FFFFFF', 'color': '#00FFFFFF'}  ####create colour for layer
        whole_area = folium.features.GeoJson(edges, name="area", style_function=lambda x: style,
                                             overlay=True)  ###create layer with whole road network
        path = folium.features.GeoJson(edges_solution, name="path", overlay=True)  ###create layer with solution edges

        nodes_to_map = nodes_solution[nodes_solution["osmid"].isin(N_supply)].reset_index()

        for i in range(0, len(nodes_to_map)):
            sources = folium.Marker(location=[nodes_to_map.loc[i, "lat"], nodes_to_map.loc[i, "lon"]],
                                    icon=folium.Icon(color='red', icon="tint"), popup="Source").add_to(m)

        sinks = folium.features.GeoJson(nodes_solution[nodes_solution["osmid"].isin(N_demand)], name="sinks",
                                        overlay=True, tooltip="Sink")

        path.add_to(m)  ###add layer to map
        whole_area.add_to(m)  ###add layer to map
        sinks.add_to(m)
        folium.LayerControl().add_to(m)###add layer control to map

        ####add labels
        folium.features.GeoJsonPopup(
            fields=['from_to', "MW", "Diameter", "Length", "Surface_type", "cost_total", 'Losses [W/m]', 'Losses [W]','Capacity_limit'],
            labels=True).add_to(path)

        # folium.features.GeoJsonPopup(fields=["osmid"], labels=True).add_to(points)
        ####save map as html#####
        m.save("TEST.html")

        ##############################################################################################
        ##############################################################################################
        ##############################################################################################
        ##############################################################################################


        ###create graph object with all solution edges
        potential_grid_area = road_nw_area.copy()

        nodes, edges = ox.graph_to_gdfs(road_nw_area)
        edges["from_to"] = "(" + edges["u"].astype(str) + ", " + edges["v"].astype(
            str) + ")"  ####create new variable in nodes where solution can be retrieved from

        edges = edges.merge(
            result_df_result[["MW", "Length", "Surface_type", "cost_total", "Diameter", 'Losses [W/m]', 'Losses [W]', 'Capacity_limit']],
            left_on=edges["from_to"], right_on=result_df_result["key_0"].astype(str), how="left")
        edges["MW"] = edges["MW"].fillna(0)

        edges.rename(
            columns={'Length': 'Pipe length [m]', 'Diameter': 'Diameter [m]', 'cost_total': 'Total costs [EUR]','Capacity_limit': "Capacity limit [MW]"},
            inplace=True)


        road_nw_area = ox.gdfs_to_graph(nodes, edges)

        edges_without_flow = [(u, v) for u, v, e in road_nw_area.edges(data=True) if e['MW'] == 0]

        for i in edges_without_flow:
            road_nw_area.remove_edge(i[0], i[1])

        nodes_not_connected = list(nx.isolates(road_nw_area))

        road_nw_area.remove_nodes_from(nodes_not_connected)

        network_solution = road_nw_area

        nodes, edges = ox.graph_to_gdfs(network_solution)
        cols_to_drop = ['key_0', 'length', 'surface_type', 'restriction', 'surface_pipe', 'existing_grid_element',
                        'inner_diameter_existing_grid_element', 'costs_existing_grid_element', 'from', 'to']
        edges = edges.drop(cols_to_drop, axis=1)
        network_solution = ox.gdfs_to_graph(nodes, edges)

        return res_sources_sinks, network_solution, potential_grid_area

