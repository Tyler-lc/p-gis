from dataclasses import dataclass
from typing import Any, Dict


def get_value(obj, key, default=None):
    try:
        return obj[key]
    except:
        print(f"[GET_VALUE] '{key}' doesn't exist using default value '{default}'")
        return default
