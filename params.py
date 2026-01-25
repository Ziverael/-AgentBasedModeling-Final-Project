from __future__ import annotations

from pydantic import BaseModel, PositiveInt, model_validator, Field, PositiveFloat, ConfigDict, NonNegativeFloat
from enum import Enum, Flag, auto
from pyvis.network import Network
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, Final
from collections import defaultdict
from itertools import combinations, product
import heapq


DAY_HOURS: Final[int] = 24


class _TransitionTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    SE: PositiveFloat = Field(le=1.0)
    I0I1: PositiveFloat = Field(le=1.0)
    I0I2: PositiveFloat = Field(le=1.0)
    I2R: PositiveFloat = Field(le=1.0)
    I2H: PositiveFloat = Field(le=1.0)
    HD: PositiveFloat = Field(le=1.0)
    HR: PositiveFloat = Field(le=1.0)

    @model_validator(mode="after")
    def check_I0_transitions(self):
        if not abs(self.I0I1 + self.I0I2 - 1.0) < 1e-9:
            raise ValueError("Sum of I0->I1 and I0->I2 must be 1.0.")
        return self
    
    @model_validator(mode="after")
    def check_I2_transitions(self):
        if not abs(self.I2H + self.I2R - 1.0) < 1e-9:
            raise ValueError("Sum of I2->R and I2->H must be 1.0.")
        return self
    
    @model_validator(mode="after")
    def check_H_transitions(self):
        if not abs(self.HD + self.HR - 1.0) < 1e-9:
            raise ValueError("Sum of H->R and H->D must be 1.0.")
        return self

TRASITION_TABLE = _TransitionTable(
    SE=0.01, # not found; choosen any value
    I0I1=0.4,
    I0I2=0.6,
    I2R=0.83,
    I2H=0.17,
    HR=0.951,
    HD=0.049,
)

class _StateAvgTimeTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    E: PositiveFloat
    I0: PositiveFloat
    I1: PositiveFloat
    I2: PositiveFloat
    H: PositiveFloat

STATE_AVG_TIMETABLE = _StateAvgTimeTable(
    E=4.0 * DAY_HOURS,
    I0=2.0 * DAY_HOURS,
    I1=7.0 * DAY_HOURS,
    I2=7.0 * DAY_HOURS,
    H=10.0 * DAY_HOURS,
)