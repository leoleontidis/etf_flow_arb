from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
