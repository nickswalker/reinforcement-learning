from math import ceil, floor
from typing import List

from rl.action import Action
from rl.state import State
from .gridworld import GridWorldState, GridWorldAction, Direction


class FeatureExtractor:
    def __init__(self):
        ()

    def num_features(self) -> int:
        raise Exception("Should've implemented this")

    def extract(self, state: State, action: Action) -> List[float]:
        raise Exception("Should've implemented this")


class DiscretizedGridWorldState(FeatureExtractor):
    def __init__(self, start_state: GridWorldState, start_action: GridWorldAction):
        self.number_of_features = len(self.extract(start_state, start_action))

    def num_features(self):
        return self.number_of_features

    def extract(self, state: GridWorldState, action: GridWorldAction) -> List[float]:
        phi = []
        phi += bin_x_y(state, 1)
        return phi


class DiscretizedGridWorldStateAction(FeatureExtractor):
    def __init__(self, start_state: GridWorldState, start_action: GridWorldAction):
        self.number_of_features = len(self.extract(start_state, start_action))

    def num_features(self):
        return self.number_of_features

    def discretized_state_action(self, state: GridWorldState, action: GridWorldAction) -> List[float]:
        phi = []
        phi += DiscretizedGridWorldState.extract(self, state)

        if action.direction == Direction.up:
            phi += [1.0]
        elif action.direction == Direction.down:
            phi += [-1.0]
        else:
            phi += [0.0]

        if action.direction == Direction.right:
            phi += [1.0]
        elif action.direction == Direction.left:
            phi += [-1.0]
        else:
            phi += [0.0]

        return phi


def bin_x_y(state: GridWorldState, square_size: int) -> List[float]:
    result = []
    map_width = len(state.map[0])
    map_height = len(state.map[1])

    binned_x = floor(state.x / square_size)
    binned_y = floor(state.y / square_size)

    max_bin_x = ceil(map_width / square_size)
    max_bin_y = ceil(map_height / square_size)
    gridding_length = max_bin_x * max_bin_y
    for x in range(0, max_bin_x):
        for y in range(0, max_bin_y):
            if binned_x == x and binned_y == y:
                result.append(1.0)
            else:
                result.append(0.0)

    return result
