from math import ceil

from .gridworld import GridWorldState, GridWorldAction, Direction


def discretized_state(state: GridWorldState) -> List[float]:
    phi = []
    phi += bin_x_y(state, 2)
    return phi


def discretized_state_action(state: GridWorldState, action: GridWorldAction) -> List[float]:
    phi = []
    phi += discretized_state(state)

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

    binned_x = state.x / square_size
    binned_y = state.y / square_size

    max_bin_x = ceil(map_width / square_size)
    max_bin_y = ceil(map_height / square_size)
    gridding_length = ceil(map_width / square_size) * ceil(map_height / square_size)
    for x in range(0, max_bin_x):
        for y in range(0, max_bin_y):
            if binned_x == x and binned_y == y:
                result.append(1.0)
            else:
                result.append(0.0)

    return result
