from enum import Enum


class Purpose(Enum):
    FITNESS = 1
    PLOTTING = 2


class Problem(Enum):
    MDVRPTW = 1
    AFVRP = 2
    AVFVRP_WITH_COOPERATIONS = 3
