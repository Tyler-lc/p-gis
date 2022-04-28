

from dataclasses import dataclass, field

from cases.exceptions.module_exception import ModuleException


@dataclass
class ModuleRuntimeException(ModuleException):
    type : str = field(default='Runtime')