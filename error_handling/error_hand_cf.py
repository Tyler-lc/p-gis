from pydantic import BaseModel, StrictFloat, StrictInt, validator, StrictStr, PositiveFloat
from typing import List

class supp_dem_dicts(BaseModel):
    id: StrictInt
    coords: List[float]
    cap: PositiveFloat

class CFData(BaseModel):
    n_supply_list: List[supp_dem_dicts]
    n_demand_list: List[supp_dem_dicts]

    @validator("n_supply_list")
    def check_n_supply_list(cls, v):
        if len(v) < 1:
            raise ValueError("There are no sources! Please be sure to choose at least one source.")
    
    @validator("n_demand_list")
    def check_n_supply_list(cls, v):
        if len(v) < 1:
            raise ValueError("There are no sinks! Please be sure to choose at least one sink.")