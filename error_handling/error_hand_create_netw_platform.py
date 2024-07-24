from pydantic import field_validator, BaseModel, StrictFloat, StrictInt, StrictStr
from typing import List


class PlatformData(BaseModel):
    network_resolution: StrictStr
    coords_list: List

    @field_validator("network_resolution")
    @classmethod
    def check_network_resolution(cls, v):
        netw_res_vals = ["high", "low", "medium_high", "medium_low"]
        if v not in netw_res_vals:
            raise ValueError(
                f"Network_resolution must be one of the following:{netw_res_vals}"
            )
        return v

    # @field_validator("ex_grid_data_json", check_fields=False)
    # @classmethod
    # # TODO complete this after existing grid data upload is implemented in the platform
    # def check_ex_grid_data(cls, v):
    #     "ex_grid data check"

    @field_validator("coords_list", check_fields=False)
    @classmethod
    # TODO: Add validation for project area after David pushes the code
    def check_polygon(cls, v):
        # polygon = [[x,y], [x,y], [x,y], [x,y]]
        if not len(v) == 4:
            raise ValueError("Length of polygon must be 4!")
