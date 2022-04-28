from pandas import to_datetime
from pydantic import BaseModel, ConstrainedFloat, StrictFloat, StrictInt, confloat, validator, StrictStr, PositiveFloat, NonNegativeFloat
from typing import List

class surface_losses(BaseModel):
    dn: PositiveFloat
    overland_losses: PositiveFloat

class PlatformData(BaseModel):
    network_resolution: StrictStr
    water_den: PositiveFloat
    factor_street_terrain: confloat(gt = 0, lt = 1)
    factor_street_overland: confloat(gt = 0, lt = 1)
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

platform_data = {
 'network_resolution': 'low',
 'water_den': 1000,
 'factor_street_terrain': 0.1,
 'factor_street_overland': 0.4,
 'heat_capacity': 4.18,
 'flow_temp': 100,
 'return_temp': 70,
 'surface_losses_dict': [{'dn': 0.02, 'overland_losses': 0.115994719393908},
  {'dn': 0.025, 'overland_losses': 0.138092834981244},
  {'dn': 0.032, 'overland_losses': 0.15109757219986},
  {'dn': 0.04, 'overland_losses': 0.171799705290563},
  {'dn': 0.05, 'overland_losses': 0.193944276611768},
  {'dn': 0.065, 'overland_losses': 0.219829984514374},
  {'dn': 0.08, 'overland_losses': 0.231572190233268},
  {'dn': 0.1, 'overland_losses': 0.241204678239951},
  {'dn': 0.125, 'overland_losses': 0.280707496411117},
  {'dn': 0.15, 'overland_losses': 0.320919871727017},
  {'dn': 0.2, 'overland_losses': 0.338510752592325},
  {'dn': 0.25, 'overland_losses': 0.326870584772369},
  {'dn': 0.3, 'overland_losses': 0.376259860034531},
  {'dn': 0.35, 'overland_losses': 0.359725182960969},
  {'dn': 0.4, 'overland_losses': 0.372648018718974},
  {'dn': 0.45, 'overland_losses': 0.427474040273953},
  {'dn': 0.5, 'overland_losses': 0.359725658523504},
  {'dn': 0.6, 'overland_losses': 0.420023799255459},
  {'dn': 0.7, 'overland_losses': 0.478951907501331},
  {'dn': 0.8, 'overland_losses': 0.540336445060049},
  {'dn': 0.9, 'overland_losses': 0.600053256925217},
  {'dn': 1.0, 'overland_losses': 0.662751592458654}],
 'ground_temp': 8,
 'ambient_temp': 25,
 'fc_dig_st': 350,
 'vc_dig_st': 700,
 'vc_dig_st_ex': 1.1,
 'fc_dig_tr': 200,
 'vc_dig_tr': 500,
 'vc_dig_tr_ex': 1.1,
 'fc_pip': 50,
 'vc_pip': 700,
 'vc_pip_ex': 1.3,
 'invest_pumps': 10000}