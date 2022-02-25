from dataclasses import dataclass
from typing import Dict


@dataclass
class KB:
    data: Dict

    def get(self, path: str = ""):
        steps = path.split(".")
        out = self.data

        try:
            for step in steps:
                out = out[step]
        except:
            out = None

        return out
