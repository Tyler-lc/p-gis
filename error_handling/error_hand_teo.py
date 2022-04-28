from pydantic import BaseModel, StrictFloat, StrictInt, validator, StrictStr
from typing import List, Dict

class in_cap(BaseModel):
    source_sink: str
    classification_type: StrictStr
    number: StrictInt

class TEOData(BaseModel):
    ex_cap_json: List[in_cap]

class TEOData2(BaseModel):
    ex_cap_json: List[Dict]