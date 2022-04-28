

from dataclasses import dataclass, field

from pydantic import ValidationError

from cases.exceptions.module_exception import ModuleException


@dataclass
class ModuleValidationException(ModuleException):
    error : ValidationError = field()