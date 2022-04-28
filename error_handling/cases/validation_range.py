from pydantic import BaseModel, ValidationError, validator

from cases.exceptions.module_validation_exception import ModuleValidationException



class ExampleModel(BaseModel):
    name : str
    age : int

    @validator('age')
    def age_must_be_above_18(cls, v):
        if v < 18:
           raise ValueError('must be higher than 18')
        return v



def run_validation_range_values():
    _dict = {
        "name" : "xpto",
        "age" : "2"
    }

    try:
        _model = ExampleModel(**_dict)

        print(_model)

    except ValidationError as e:
        raise ModuleValidationException(code=1, msg="Problemd with ExampleModel", error=e)
