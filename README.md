# EMB3RS Geographical Information System (GIS) Module
## EMb3RS Project
EMB3Rs (“User-driven Energy-Matching & Business Prospection Tool for Industrial Excess Heat/Cold Reduction, Recovery and Redistribution) is a European project funded under the H2020 Program (Grant Agreement N°847121) to develop an open-sourced tool to match potential sources of excess thermal energy with compatible users of heat and cold.

Users, like industries and other sources that produce excess heat, will provide the essential parameters, such as their location and the available excess thermal energy. The EMB3Rs platform will then autonomously and intuitively assess the feasibility of new business scenarios and identify the technical solutions to match these sources with compatible sinks. End users such as building managers, energy communities or individual consumers will be able to determine the costs and benefits of industrial excess heat and cold utilization routes and define the requirements for implementing the most promising solutions.

The EMB3Rs platform will integrate several analysis modules that will allow a full exploration of the feasible technical routes to the recovery and use of the available excess thermal energy.

## Module Overview
The Geographical Information System (GIS) modules' purpose within the EMB3RS platform is to analyze the network dimension and bring in the spatial dimension between sources and sinks. The application of the GIS is tailored for looking into the option of reusing the excess heat/cold at a certain distance within a District Heating and Cooling (DHC) system. It assumes a potential network solution between a particular set of sources and sinks among the Open Street Map (OSM) road network. The related investment costs into the grid and the corresponding heat/cold losses are calculated based on that network solution.

The GIS module receives information from the core functionalities (CF) module, the knowledge base (KB) as well as the platform/user and sends information to the other calculation modules, namely the CF module, Techno-Economic Optimization (TEO) module, Market Module (MM), and Business Module (BM).

The main features of the GIS module calculations are:
- DHC network calculation based on different heat/cold sources and sinks (routing),
- calculation of the heat/cold losses and investment costs of the resulting DHC network solution (heat loss and cost calculation).

## General Module Architecture
The GIS module consists of two main functions: "create_network" and "optimize_network".

The "create_network" function serves as the first step in the GIS module. It receives inputs from the user/platform, the CF module, and the TEO module -starting from the second iteration if the advanced calculation is chosen-. Then, it returns an Open Street Map graph to the platform.

The "optimize_network" function is the second step of the GIS module. It calculates a thermal network solution and related thermal losses and investment costs.

## Module Requirement
The integrated version of the GIS module has following dependencies:
-	python = 3.9
-	osmnx = 1.1.2
-	scikit-learn = 1.0.2
-	numpy = 1.22.3
-	pyomo =5.7
-	haversine = 2.5.1
-	pandas = 1.4.1
-	folium = 0.12.1
-	geopandas = 0.10.2
-	shapely = 1.8.0
-	networkx = 2.7.1
-	colorama = 0.4.4
-	jsonpickle = 2.1.0
-	gurobipy = 9.5.1.
-	pydantic 

Please note that for the package "gurobipy", the channel "pip" is used. For all the other remaining dependencies "conda-forge" channel is used. Also, the use of an up-to-date package version is recommended. However, the user should be aware that any function used in the model might be deprecated. The user should use the same versions given above, and is encouraged to report any issues found.

The optimization models are modeled with PYOMO and solved with the GUROBI solver. Therefore, a valid GUROBI license is required.
