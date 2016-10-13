from enum import Enum
from typing import List, Set

from rl.action import Action
from rl.domain import Domain
from rl.state import State
from rl.task import Task


class Direction(Enum):
    up = 0
    right = 1
    down = 2
    left = 3

    def __int__(self):
        return self.value

    def __str__(self):
        if self == Direction.up:
            return "up"
        elif self == Direction.right:
            return "right"
        elif self == Direction.bottom:
            return "bottom"
        elif self == Direction.left:
            return "left"


class GridItem(Enum):
    empty = 0
    pit = 1
    exit = 2


class GridWorldState(State):
    def __init__(self, x: int, y: int, map: List[List[int]]):
        self.x = x
        self.y = y
        self.map = map

    def __hash__(self):
        return len(self.map) * self.y + self.x

    def __eq__(self, other):
        # Assumes two states have the same map!
        if isinstance(other, GridWorldState):
            return other.__hash__() == self.__hash__()
        return False

    def __str__(self):
        result = "_" * len(self.map[0]) * 3
        result += "\n"
        for y in range(0, len(self.map)):
            for x in range(0, len(self.map[0])):
                item = self.map[y][x]
                if x == self.x and y == self.y:
                    result += "A"
                elif item == GridItem.empty:
                    result += " "
                elif item == GridItem.exit:
                    result += "X"
                result += " | "
            result += "\n"
        result += "_" * len(self.map[0]) * 3
        result += "\n"
        return result


class GridWorldAction(Action):
    def __init__(self, direction: Direction):
        super().__init__()
        self.direction = direction

    def __hash__(self):
        return self.direction.__hash__()

    def __eq__(self, other):
        # Assumes two states have the same map!
        if isinstance(other, GridWorldAction):
            return other.__hash__() == self.__hash__()
        return False

    def __str__(self):
        return self.direction.__str__()


class GridWorld(Domain):
    def __init__(self, width: int, height: int):
        self.map = [[0] * height for _ in range(0, width)]
        self.width = width
        self.height = height
        self.agent_x = 0
        self.agent_y = 0
        self.actions = [GridWorldAction(Direction.up), GridWorldAction(Direction.right),
                        GridWorldAction(Direction.down), GridWorldAction(Direction.left)]

    def get_actions(self, state: State) -> Set[Action]:
        return {GridWorldAction(Direction.up), GridWorldAction(Direction.right), GridWorldAction(Direction.down),
                GridWorldAction(Direction.left)}

    def apply_action(self, action: Action):
        assert isinstance(action, GridWorldAction)
        # Move the agent
        if action.direction is Direction.up:
            self.agent_y += 1
        elif action.direction is Direction.right:
            self.agent_x += 1
        elif action.direction is Direction.down:
            self.agent_y -= 1
        elif action.direction is Direction.left:
            self.agent_x -= 1

        # Clamp
        if self.agent_x < 0:
            self.agent_x = 0
        if self.agent_y < 0:
            self.agent_y = 0

        if self.agent_x >= self.width:
            self.agent_x = self.width - 1
        if self.agent_y >= self.height:
            self.agent_y = self.height - 1

    def get_current_state(self) -> GridWorldState:
        return GridWorldState(self.agent_x, self.agent_y, self.map)

    def reset(self):
        self.agent_x = 0
        self.agent_y = 0

    def place_exit(self, x: int, y: int):
        self.map[y][x] = GridItem.exit


class ReachExit(Task):
    def reward(self, state, action, state_prime) -> float:
        if state.x == state_prime.x and state.y == state_prime.y:
            return -5
        elif self.domain.map[state_prime.y][state_prime.x] == GridItem.exit:
            return 20
        else:
            return -1

    def stateisfinal(self, state) -> bool:
        item = self.domain.map[state.y][state.x]
        return item == GridItem.exit or item == GridItem.pit
