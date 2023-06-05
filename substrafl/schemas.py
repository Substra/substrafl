from enum import Enum


class TaskType(str, Enum):
    INITIALIZATION = "init"
    TRAIN = "train"
    AGGREGATE = "aggregate"
