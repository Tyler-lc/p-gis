from pandas import to_datetime
from pydantic import (
    field_validator,
    Field,
    BaseModel,
    StrictStr,
    PositiveFloat,
    NonNegativeFloat,
)
from typing import List
from typing_extensions import Annotated


class surface_losses(BaseModel):
    dn: PositiveFloat
    overland_losses: PositiveFloat


class PlatformData(BaseModel):
    network_resolution: StrictStr
    water_den: PositiveFloat
    factor_street_terrain: Annotated[float, Field(gt=0, lt=1)]
    factor_street_overland: Annotated[float, Field(gt=0, lt=1)]
    heat_capacity: PositiveFloat
    flow_temp: PositiveFloat
    return_temp: PositiveFloat
    ground_temp: PositiveFloat
    ambient_temp: PositiveFloat
    fc_dig_st: NonNegativeFloat
    vc_dig_st: NonNegativeFloat
    vc_dig_st_ex: NonNegativeFloat
    fc_dig_tr: NonNegativeFloat
    vc_dig_tr: NonNegativeFloat
    vc_dig_tr_ex: NonNegativeFloat
    fc_pip: NonNegativeFloat
    vc_pip: NonNegativeFloat
    vc_pip_ex: NonNegativeFloat
    invest_pumps: NonNegativeFloat

    surface_losses_dict: List[surface_losses]

    @field_validator("network_resolution")
    @classmethod
    def check_network_resolution(cls, v):
        netw_res_vals = ["high", "low", "medium_high", "medium_low"]
        if v not in netw_res_vals:
            raise ValueError(
                f"Network_resolution must be one of the following:{netw_res_vals}"
            )
        return v

    @field_validator("ex_grid_data_json", check_fields=False)
    @classmethod
    # TODO complete this after existing grid data upload is implemented in the platform
    def check_ex_grid_data(cls, v):
        "ex_grid data check"
