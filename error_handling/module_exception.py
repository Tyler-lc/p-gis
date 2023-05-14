from dataclasses import dataclass, field


@dataclass
class ModuleException(Exception):
    code: str = field()
    msg: str = field()

    def __str__(self):
        return self.msg
