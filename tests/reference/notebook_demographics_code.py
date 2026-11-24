"""VERBATIM demographic cleaning from notebook cell 10 (reference oracle, do not edit).
`meta` has columns age, sex, height, weight."""
import numpy as np
import pandas as pd

def notebook_demographics(meta):
    demo = pd.DataFrame(index=meta.index)
    demo["age"] = meta["age"].where(meta["age"] <= 120, 90)
    demo["sex"] = meta["sex"]                                   # 0 = female, 1 = male in PTB-XL
    demo["height"] = meta["height"].where(meta["height"].between(50, 230))
    demo["weight"] = meta["weight"].where(meta["weight"].between(3, 250))
    bmi = demo["weight"] / (demo["height"] / 100) ** 2
    demo["bmi"] = bmi.where(bmi.between(8, 70))
    
    return demo
