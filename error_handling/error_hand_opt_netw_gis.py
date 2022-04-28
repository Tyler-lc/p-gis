from pandas import to_datetime
from pydantic import BaseModel, ConstrainedFloat, StrictFloat, StrictInt, validator, StrictStr, PositiveFloat, NonNegativeFloat
from typing import List

class node_dict(BaseModel):
    y: float
    x: float
    osmid: int
    lon: float
    lat: float

class edge_dict(BaseModel):
    length: PositiveFloat
    surface_pipe: StrictInt
    inner_diameter_existing_grid_element: NonNegativeFloat
    costs_existing_grid_element: NonNegativeFloat
    surface_type: StrictStr
    restriction: StrictInt
    existing_grid_element: StrictInt
    #from: int
    to: int

class Gisdata(BaseModel):
    demand_list: List[StrictInt]
    supply_list: List[StrictInt]
    nodes: List[node_dict]
    edges: List[edge_dict]

    @validator("supply_list")
    def check_supply_list(cls, v):
        if len(v)<1:
            raise ValueError("There is no source to optimize!")

    @validator("demand_list")
    def check_demand_list(cls, v):
        if len(v)<1:
            raise ValueError("There is no sink to optimize!")

gis_data = {
    "supply_list": [1],
    "demand_list": [3, 5, 4],
    "nodes":[{'y': 4292305.7501438325,
  'x': 491558.94980194967,
  'osmid': 100,
  'lon': -9.09718,
  'lat': 38.77944},
 {'y': 4292346.709711878,
  'x': 491651.9331066382,
  'osmid': 200,
  'lon': -9.09611,
  'lat': 38.77981}],
    "edges": [
  {'length': 120.0,
  'surface_pipe': 0,
  'inner_diameter_existing_grid_element': 0.3,
  'costs_existing_grid_element': 10000,
  'surface_type': 'None',
  'restriction': 0,
  'existing_grid_element': 1,
  'from': 100,
  'to': 200},
 {'length': 120.0,
  'surface_pipe': 1,
  'inner_diameter_existing_grid_element': 0.05,
  'costs_existing_grid_element': 5000,
  'surface_type': 'None',
  'restriction': 0,
  'existing_grid_element': 1,
  'from': 200,
  'to': 300}]
  }
