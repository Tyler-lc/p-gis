# EMB3Rs Geographical Information System (GIS) Module

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## EMB3RS Project

EMB3Rs (“User-driven Energy-Matching & Business Prospection Tool for Industrial Excess Heat/Cold Reduction, Recovery and Redistribution) is a European project funded under the H2020 Program (Grant Agreement N°847121) to develop an open-source tool to match potential sources of excess thermal energy with compatible users of heat and cold.

Users, such as industries and other sources that produce excess heat, will provide the essential parameters, such as their location and the available excess thermal energy. The EMB3Rs platform will then autonomously and intuitively assess the feasibility of new business scenarios and identify the technical solutions to match these sources with compatible sinks. End users such as building managers, energy communities, or individual consumers will be able to determine the costs and benefits of industrial excess heat and cold utilization routes and define the requirements for implementing the most promising solutions.

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

The general model architecture is given in below:
![image](https://user-images.githubusercontent.com/98012853/165799218-3486110b-2010-4b05-b859-74f4dacd6624.png)

## Module Requirements

The integrated version of the GIS module has following dependencies:

- python = 3.9
- osmnx = 1.1.2
- scikit-learn = 1.0.2
- numpy = 1.22.3
- pyomo =5.7
- haversine = 2.5.1
- pandas = 1.4.1
- folium = 0.12.1
- geopandas = 0.10.2
- shapely = 1.8.0
- networkx = 2.7.1
- colorama = 0.4.4
- jsonpickle = 2.1.0
- gurobipy = 9.5.1.
- pydantic

Please note that for the package "gurobipy", the channel "pip" is used. For all the other remaining dependencies "conda-forge" channel is used. Also, the use of an up-to-date package version is recommended. However, the user should be aware that any function used in the model might be deprecated. The user should use the identical versions above and be encouraged to report any issues.

The optimization models are modeled with PYOMO and solved with the GUROBI solver. Therefore, a valid GUROBI license is required.

## Inputs and Outputs

The overall input and output structure of the GIS module is given in below:
![image](https://user-images.githubusercontent.com/98012853/165799907-19c696ee-67e0-491d-a89e-9dfe957fc62c.png)

Under the “Function” column, it is indicated which function is using the input. Information on which input belongs to which function is not relevant to the users but to developers. It is also shown if the input is mandatory or not. Please note that all mandatory inputs except for “Project Area” have a default value stored in the Knowledge Base. In other words, if the user does not have enough information to set a value for those variables or basically wants to use default variables, they have the option not to give input. However, the user must provide the “Project Area” input by choosing it via the platform.

The inputs (their labels) that are expected from the user, descriptions of the inputs, and their units are given below:
| **Input Label** | **Function** | **Mandatory** | **Description** | **Unit** |
|:---: |:---: |:---: |:---: |:---: |
| Network Resolution | create_network | TRUE | Defines if network resolution is high or low, i.e., how detailed the streets are loaded. If a large network is used, network resolution should be set to low to decrease computational time. Set to high by default. | - |
| Existing Grid Network | create_network | FALSE | The information on the existing grid network. For each pipe, IDs of sources/sinks connected by the pipe, latitudes, and longitudes of those sources/sinks, diameter and length of the pipe, total cost of the pipe, and if the respective pipe is a surface pipe should be defined. | Diameter in m. Length in m. Total cost of the pipe in EUR. |
| Project Area | create_network | TRUE | The area that will be considered for the grid. User could specify the area by drawing a rectangular shape on the map via platform. | - |
| Investment Costs for Pumps | optimize_network | FALSE | Investment costs for pumps. Set to 0 by default. | EUR |
| Fixed Digging Cost for Street | optimize_network | TRUE | Fixed digging cost for streets. Set to 350 by default. | EUR/m |
| Variable Digging Cost for Street | optimize_network | TRUE | Variable digging cost for streets. Set to 700 by default. | EUR/m² |
| Exponent Street | optimize_network | TRUE | The exponent of the digging cost for the street. Set to 1.1 by default. | - |
| Fixed Digging Cost for Terrain | optimize_network | TRUE | Fixed digging cost for terrains. Set to 200 by default. | EUR/m |
| Variable Digging Cost for Terrain | optimize_network | TRUE | Variable digging cost for terrains. Set to 500 by default. | EUR/m² |
| Exponent Terrain | optimize_network | TRUE | The exponent of the digging cost for the terrain. Set to 1.1 by default. | - |
| Average Ambient Temperature | optimize_network | TRUE | Yearly average ambient temperature. Set to 25 by default. | °C |
| Average Ground Temperature | optimize_network | TRUE | Yearly average ground temperature. Set to 8 by default. | °C |
| Average Flow Temperature | optimize_network | TRUE | Yearly average flow temperature. Set to 100 by default. | °C |
| Average Return Temperature | optimize_network | TRUE | Yearly average return temperature. Set to 70 by default. | °C |
| Heat Capacity | optimize_network | TRUE | Heat capacity at a specific temperature (average of flow and return temperatures). Set to 4.18 by default. | J/kgK |
| Water Density | optimize_network | TRUE | Water density at a specific temperature (average of flow and return temperatures). Set to 1000 by default. | kg/m3 |
| Fixed Piping Cost | optimize_network | TRUE | The fixed component of the piping cost. Set to 50 by default. | EUR/m |
| Variable Piping Cost | optimize_network | TRUE | The fixed component of the piping cost. Set to 700 by default. | EUR/m² |
| Exponent Piping | optimize_network | TRUE | The exponent of the piping cost. Set to 1.3 by default. | - |
| Cost Factor Street vs. Terrain | optimize_network | TRUE | Determines how much cheaper it is to lay 1 m of pipe into a terrain than a street. Expressed in decimals: 0.1 means it is 10% cheaper. | Decimals |
| Cost Factor Street vs. Overland | optimize_network | TRUE | Determines how much cheaper it is to place 1 m of the pipe over the ground than putting it into the street. Expressed in decimals: 0.4 means it is 40% cheaper. | Decimals |

Also, note that the unit digging and piping costs are calculated in the following format:
<img src="https://render.githubusercontent.com/render/math?math=Unit Digging/Piping\ Costs[EUR/m] = fixed cost + [(diameter)(variable cost)]^{exponent}">

Therefore, all the inputs named as a fixed cost in the table above correspond to the fixed cost in the formula above. Similarly, inputs named as a variable cost correspond to the variable cost in the formula above. Finally, the inputs named as the exponent correspond to the exponent in the formula above. The model calculates the diameter, so it is not user input. If a pipe is an overland pipe, the model automatically assigns a digging cost of zero to it.

The outputs of the GIS Module are

- the network solution visualized on Open Street Map,
- the network losses,
- the investment costs.

The GIS module also outputs the potential grid area independent of the network solution. This potential grid area shows all the possible routes for the pipes on OSM.

## License

Copyright 2022 Ali Kök

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Acknowledgments

The EMB3RS project has received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement No 847121. This publication reflects only the views of its authors, and the European Commission cannot be held responsible for its content.

TEST CHANGE
ANOTHER CHANGE
