from pydantic import field_validator, BaseModel, StrictFloat, StrictInt, StrictStr
from typing import List, Dict, Union

class in_cap(BaseModel):
    source_sink: Union[str, None] = None
    classification_type: StrictStr
    number: StrictInt

class TEOData(BaseModel):
    ex_cap: List[in_cap]

class TEOData2(BaseModel):
    ex_cap: list
    
    @field_validator('ex_cap')
    @classmethod
    def validateExCapStruct(cls, v):
        # print(v)
        for item in v:
            col_name = list(item)
            if (
                not (col_name[0] == 'source_sink') or
                not (col_name[1] == 'classification_type') or
                not (col_name[2] == 'number')
            ):
                raise ValueError(
                    "Structure of Ex Cap is Not Correct e.g. : ['source_sink', 'classification_type', 'number', '<timestamps>' (...)")