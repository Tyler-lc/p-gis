from pydantic import BaseModel, StrictFloat, StrictInt, validator, StrictStr
from typing import List

class PlatformData(BaseModel):
    network_resolution: StrictStr
    polygon: List

    @validator('network_resolution')
    def check_network_resolution(cls, v):
        netw_res_vals = ["high", "low"]
        if v not in netw_res_vals:
           raise ValueError(f'Network_resolution must be one of the following:{netw_res_vals}')
        return v

    @validator('ex_grid_data_json', check_fields=False)
    # TODO complete this after existing grid data upload is implemented in the platform
    def check_ex_grid_data(cls, v):
        "ex_grid data check"

    @validator('polygon', check_fields=False)
    #TODO: Add validation for project area after David pushes the code
    def check_polygon(cls, v):
        #polygon = [[x,y], [x,y], [x,y], [x,y]]
        if not len(v) == 4:
            raise ValueError("Length of polygon must be 4!")