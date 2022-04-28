from pydantic import BaseModel, ValidationError, validator

from cases.exceptions.module_validation_exception import ModuleValidationException



class ExampleModel(BaseModel):  # STRUCTURE VALIDATION
    name : str                  # Mandatory
    age : int                   # Mandatory
    num_dogs : int              # Mandatory
    dog_names : list[str]       # Mandatory

    md : str = None             # Optional
    random_pro : str = None     # Optional

    @validator('dog_names')
    def dog_names_must_be_same_as_length(cls, v, values, **kwargs):
        if values['num_dogs'] != len(v):
            raise ValueError("Doesn't make sense")
        return v

    @validator('md')
    def check_md_stuff(cls,v, values, **kwargs):
        if v == 'community':
            if values["random_pro"] != 'p2p':
                raise ValueError("if Community can't be P2P")
        return v


def run_validation_conditional_values():
    _dict = {
        "name" : "xpto",
        "age" : 1,
        "num_dogs" : 1,
        "dog_names" : ["doggy"]
    }

    try:
        _model = ExampleModel(**_dict)

        print(_model.schema_json(indent=2))

    except ValidationError as e:
        raise ModuleValidationException(code=1, msg="Problem with ExampleModel", error=e)
    except Exception as e:
        print(e)