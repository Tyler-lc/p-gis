import json

import pandas as pd
from functions.create_network import create_network, run_create_network
from functions.optimize_network import optimize_network, run_optimize_network
from utilities.create_ex_grid import create_ex_grid
import warnings

warnings.filterwarnings("ignore")

n_supply_list = [
    {"id": 2, "coords": [38.77988, -9.10126], "cap": 3},
    {"id": 1, "coords": [38.76992, -9.1014], "cap": 35},
    # {"id": 6, "coords": [38.75246, -9.23775], "cap": 35},
]

n_demand_list = [
    {"id": 3, "coords": [38.77387, -9.1003], "cap": 0.365},
    {"id": 5, "coords": [38.7764, -9.1019], "cap": 0.502},
    {"id": 4, "coords": [38.77438, -9.10429], "cap": 2.113},
    # {"id": 4, "coords": [38.76992, -9.1014], "cap": 2.113}, #this demand point has same location with a supply point fot test purposes
    # {"id": 6, "coords": [38.77963,-9.09487],"cap": 2.113}
]

ex_grid_data_json = [
    {
        "from": 100,
        "lon_from": -9.09718,
        "lat_from": 38.77944,
        "to": 200,
        "lon_to": -9.09611,
        "lat_to": 38.77981,
        "dn": 0.3,
        "total_costs": 10000,
        "length": 120,
        "surface_pipe": 0,
    },
    {
        "from": 200,
        "lon_from": -9.09611,
        "lat_from": 38.77981,
        "to": 300,
        "lon_to": -9.0956,
        "lat_to": 38.77895,
        "dn": 0.05,
        "total_costs": 5000,
        "length": 120,
        "surface_pipe": 1,
    },
    {
        "from": 300,
        "lon_from": -9.0956,
        "lat_from": 38.77895,
        "to": 400,
        "lon_to": -9.09519,
        "lat_to": 38.77825,
        "dn": 0.5,
        "total_costs": 0,
        "length": 10,
        "surface_pipe": 0,
    },
    {
        "from": 400,
        "lon_from": -9.09519,
        "lat_from": 38.77825,
        "to": 500,
        "lon_to": -9.09505,
        "lat_to": 38.77813,
        "dn": 0.1,
        "total_costs": 500,
        "length": 300,
        "surface_pipe": 0,
    },
]

# ex_cap_json is a list not a dict
ex_cap_json = [
    {
        "source_sink": "residential_buildings",
        "classification_type": "sink",
        "number": 5,
        "2008": 893.1349258904,
        "3285": 893.1349258904,
        "6205": 893.1349258904,
        "6570": 893.1349258904,
        "7118": 893.1349258904,
        "183": 0.0,
        "6753": 0.0,
        "5110": 0.0,
        "1825": 638.2094571233,
        "2190": 135.2882810137,
        "2373": 593.9605489041,
        "2555": 371.4467658904,
        "2738": 0.0,
        "2920": 251.7572991644,
        "3103": 0.0,
        "3468": 202.8833073151,
        "365": 341.1086554247,
        "3650": 566.5859164384,
        "3833": 548.8782638356,
        "4015": 339.6530151096,
        "4198": 306.0696772192,
        "4380": 315.6111272329,
        "5293": 888.7989820548,
        "548": 3.5555012644,
        "5840": 649.0522260274,
        "6935": 103.0706465753,
        "730": 278.7728724795,
        "7300": 445.5777458904,
        "7665": 343.8486944795,
        "7848": 99.7500714384,
        "8213": 498.3566949315,
        "913": 13.8270229616,
        "1095": 553.7654519178,
        "1278": 619.0890836986,
        "1460": 673.0445313699,
        "1643": 683.0983791781,
    },
    {
        "source_sink": "office_buildings",
        "classification_type": "sink",
        "number": 4,
        "2008": 0.0,
        "3285": 0.0,
        "6205": 0.0,
        "6570": 0.0,
        "7118": 0.0,
        "183": 893.1349258904,
        "6753": 893.1349258904,
        "5110": 1.5211813295,
        "1825": 0.0,
        "2190": 22.2166762041,
        "2373": 46.7767907945,
        "2555": 383.8066716438,
        "2738": 469.0463745205,
        "2920": 0.0,
        "3103": 203.1800539041,
        "3468": 0.0,
        "365": 552.0262491781,
        "3650": 0.0,
        "3833": 62.0358041781,
        "4015": 0.0,
        "4198": 21.5411939151,
        "4380": 31.0841415164,
        "5293": 4.3359133233,
        "548": 132.915241411,
        "5840": 244.0826821233,
        "6935": 0.0,
        "730": 19.3570210151,
        "7300": 447.5571445205,
        "7665": 0.0,
        "7848": 0.0,
        "8213": 0.0,
        "913": 53.7752794658,
        "1095": 36.9569796438,
        "1278": 32.1001247192,
        "1460": 23.5848277384,
        "1643": 4.8846615904,
    },
    {
        "source_sink": "hotel",
        "classification_type": "sink",
        "number": 3,
        "2008": 0.0,
        "3285": 0.0,
        "6205": 0.0,
        "6570": 0.0,
        "7118": 0.0,
        "183": 0.0,
        "6753": 0.0,
        "5110": 891.6137443836,
        "1825": 254.9254474795,
        "2190": 735.6299654795,
        "2373": 252.3975684521,
        "2555": 137.8814670685,
        "2738": 424.0885158904,
        "2920": 641.377630274,
        "3103": 689.9548542466,
        "3468": 690.2516043836,
        "365": 0.0,
        "3650": 326.5489810685,
        "3833": 282.220840137,
        "4015": 553.481900137,
        "4198": 565.5240519178,
        "4380": 546.4396546575,
        "5293": 0.0,
        "548": 756.6641715068,
        "5840": 0.0,
        "6935": 790.0642793151,
        "730": 595.0050284932,
        "7300": 0.0,
        "7665": 549.2862065753,
        "7848": 793.3848367123,
        "8213": 394.7782309589,
        "913": 825.5325908219,
        "1095": 302.4124978767,
        "1278": 241.9456873151,
        "1460": 196.5055423014,
        "1643": 205.1518634795,
    },
    {
        "source_sink": "metal_casting",
        "classification_type": "source",
        "number": 1,
        "2008": 893.1349172603,
        "3285": 893.1349172603,
        "6205": 893.1349172603,
        "6570": 893.1349172603,
        "7118": 893.1349172603,
        "183": 893.1349172603,
        "6753": 893.1349172603,
        "5110": 893.1349172603,
        "1825": 893.1349172603,
        "2190": 893.1349172603,
        "2373": 893.1349172603,
        "2555": 893.1349172603,
        "2738": 893.1349172603,
        "2920": 893.1349172603,
        "3103": 893.1349172603,
        "3468": 893.1349172603,
        "365": 893.1349172603,
        "3650": 893.1349172603,
        "3833": 893.1349172603,
        "4015": 893.1349172603,
        "4198": 893.1349172603,
        "4380": 893.1349172603,
        "5293": 893.1349172603,
        "548": 893.1349172603,
        "5840": 893.1349172603,
        "6935": 893.1349172603,
        "730": 893.1349172603,
        "7300": 893.1349172603,
        "7665": 893.1349172603,
        "7848": 893.1349172603,
        "8213": 893.1349172603,
        "913": 893.1349172603,
        "1095": 893.1349172603,
        "1278": 893.1349172603,
        "1460": 893.1349172603,
        "1643": 893.1349172603,
    },
]

coords = [
    [38.77988, -9.10126],
    [38.76992, -9.1014],
    [38.75246, -9.23775],
    [38.77963, -9.09487],
]

ex_grid = create_ex_grid(ex_grid_data_json)

network_resolution = "low"

input_data = {
    "platform": {
        "n:_supply_list": n_supply_list,
        "n_demand_list": n_demand_list,
        "ex_grid": ex_grid_data_json,
        "ex_cap": ex_cap_json,
        "network_resolution": network_resolution,
        "coords_list": coords,
    },
    "cf-module": {"n_supply_list": n_supply_list, "n_demand_list": n_demand_list},
    "teo-module": {"ex_cap": ex_cap_json},
}

from utilities import kb, kb_data

create_output = run_create_network(input_data, kb_data.kb)

################################################################################
######GRID SPECIFIC INFORMATION#################################################

###########INVESTMENT COSTS PUMPS###############

invest_pumps = 10000

# invest_pumps = json.dumps(invest_pumps)

###########COSTS DIGGING STREET#################
fc_dig_st = 350
vc_dig_st = 700

# fc_dig_st = json.dumps(fc_dig_st)
# vc_dig_st = json.dumps(vc_dig_st)

###########COSTS DIGGING TERRAIN#################
fc_dig_tr = 200
vc_dig_tr = 500

# fc_dig_tr = json.dumps(fc_dig_tr)
# vc_dig_tr = json.dumps(vc_dig_tr)

###########COSTS PIPES###########################
fc_pip = 50
vc_pip = 700

# fc_pip = json.dumps(fc_pip)
# vc_pip = json.dumps(vc_pip)

###########COST FORMULAS EXPONENTS###############
vc_dig_st_ex = 1.1
vc_dig_tr_ex = 1.1
vc_pip_ex = 1.3

# vc_dig_st_ex = json.dumps(vc_dig_st_ex)
# vc_dig_tr_ex = json.dumps(vc_dig_tr_ex)
# vc_pip_ex = json.dumps(vc_pip_ex)

#####COST DIFFERENCE FACTOR STREET/TERRAIN &
# STREET/OVERLAND#######

factor_street_terrain = 0.10
factor_street_overland = 0.4

# factor_street_terrain = json.dumps(factor_street_terrain)
# factor_street_overland = json.dumps(factor_street_overland)

###########GRID TEMPERATURES/WATER FEATURES/HEAT CAP.####

flow_temp = 100
return_temp = 70
water_den = 1000
heat_capacity = 4.18

# flow_temp = json.dumps(flow_temp)
# return_temp = json.dumps(return_temp)
# water_den = json.dumps(water_den)
# heat_capacity = json.dumps(heat_capacity)

###########ENVIRONMENTAL INPUTS#################

ground_temp = 8
ambient_temp = 25

# ground_temp = json.dumps(ground_temp)
# ambient_temp = json.dumps(ambient_temp)

###########DATAFRAME FOR SURFACE LOSSES#################

surface_losses_json = [
    {"dn": 0.02, "overland_losses": 0.115994719393908},
    {"dn": 0.025, "overland_losses": 0.138092834981244},
    {"dn": 0.032, "overland_losses": 0.15109757219986},
    {"dn": 0.04, "overland_losses": 0.171799705290563},
    {"dn": 0.05, "overland_losses": 0.193944276611768},
    {"dn": 0.065, "overland_losses": 0.219829984514374},
    {"dn": 0.08, "overland_losses": 0.231572190233268},
    {"dn": 0.1, "overland_losses": 0.241204678239951},
    {"dn": 0.125, "overland_losses": 0.280707496411117},
    {"dn": 0.15, "overland_losses": 0.320919871727017},
    {"dn": 0.2, "overland_losses": 0.338510752592325},
    {"dn": 0.25, "overland_losses": 0.326870584772369},
    {"dn": 0.3, "overland_losses": 0.376259860034531},
    {"dn": 0.35, "overland_losses": 0.359725182960969},
    {"dn": 0.4, "overland_losses": 0.372648018718974},
    {"dn": 0.45, "overland_losses": 0.427474040273953},
    {"dn": 0.5, "overland_losses": 0.359725658523504},
    {"dn": 0.6, "overland_losses": 0.420023799255459},
    {"dn": 0.7, "overland_losses": 0.478951907501331},
    {"dn": 0.8, "overland_losses": 0.540336445060049},
    {"dn": 0.9, "overland_losses": 0.600053256925217},
    {"dn": 1.0, "overland_losses": 0.662751592458654},
]

input_optimization_data = {
    "gis-module": {
        "nodes": create_output["nodes"],
        "edges": create_output["edges"],
        "demand_list": create_output["demand_list"],
        "supply_list": create_output["supply_list"],
    },
    "platform": {
        "ex_grid": ex_grid_data_json,
        "network_resolution": network_resolution,
        "water_den": water_den,
        "factor_street_terrain": factor_street_terrain,
        "factor_street_overland": factor_street_overland,
        "heat_capacity": heat_capacity,
        "flow_temp": flow_temp,
        "return_temp": return_temp,
        "surface_losses_dict": surface_losses_json,
        "ground_temp": ground_temp,
        "ambient_temp": ambient_temp,
        "fc_dig_st": fc_dig_st,
        "vc_dig_st": vc_dig_st,
        "vc_dig_st_ex": vc_dig_st_ex,
        "fc_dig_tr": fc_dig_tr,
        "vc_dig_tr": vc_dig_tr,
        "vc_dig_tr_ex": vc_dig_tr_ex,
        "fc_pip": fc_pip,
        "vc_pip": vc_pip,
        "vc_pip_ex": vc_pip_ex,
        "invest_pumps": invest_pumps,
    },
    "cf-module": {"n_supply_list": n_supply_list, "n_demand_list": n_demand_list},
    "teo-module": {"ex_cap": ex_cap_json},
}

optimize_output = run_optimize_network(input_optimization_data, kb_data.kb)
