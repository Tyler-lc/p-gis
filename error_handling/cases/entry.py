

from cases.exceptions.module_runtime_exception import ModuleRuntimeException
from cases.exceptions.module_validation_exception import ModuleValidationException
from cases.models.platform import PlatformModel
from cases.runtime import run_runtime
from cases.validation_conditional import run_validation_conditional_values
from cases.validation_range import run_validation_range_values


def run():
    input_data = {
        "platform": {
            "group_of_sinks": [
                {
                    "id": 1,
                    "name": "Sink Demo",
                    "consumer_type" : "household"
                }
            ]
        }
    }

    platform = PlatformModel(**input_data["platform"])
    print(platform)
