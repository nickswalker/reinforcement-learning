import random
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
        elif self == Direction.down:
            return "down"
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
        return str(self.x) + ", " + str(self.y)
        result = "_" * len(self.map[0]) * 3
        result += "\n"
        for y in reversed(range(0, len(self.map))):
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
    def __init__(self, width: int, height: int, agent_x_start: int, agent_y_start: int, wind=False,
                 wind_strengths=None):
        self.map = [[0] * width for _ in range(0, height)]
        self.width = width
        self.height = height
        self.agent_x = agent_x_start
        self.agent_y = agent_y_start
        self.actions = [GridWorldAction(Direction.up), GridWorldAction(Direction.right),
                        GridWorldAction(Direction.down), GridWorldAction(Direction.left)]
        self.wind = wind
        self.wind_strengths = wind_strengths
        self.agent_start_x = agent_x_start
        self.agent_start_y = agent_y_start
        if self.wind:
            assert len(wind_strengths) == width

    def get_actions(self, state: State) -> Set[Action]:
        return {GridWorldAction(Direction.up), GridWorldAction(Direction.right), GridWorldAction(Direction.down),
                GridWorldAction(Direction.left)}

    def apply_action(self, action: Action):
        assert isinstance(action, GridWorldAction)

        if self.wind:
            strength = self.wind_strengths[self.agent_x]
            if strength > 0:
                die_roll = random.randint(0, 2)
                if die_roll == 0:
                    strength -= 1
                elif die_roll == 1:
                    strength += 0
                elif die_roll == 2:
                    strength += 1

        # Move the agent
        if action.direction is Direction.up:
            self.agent_y += 1
        elif action.direction is Direction.right:
            self.agent_x += 1
        elif action.direction is Direction.down:
            self.agent_y -= 1
        elif action.direction is Direction.left:
            self.agent_x -= 1

        if self.wind:
            self.agent_y += strength

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
        self.agent_x = self.agent_start_x
        self.agent_y = self.agent_start_y

    def place_exit(self, x: int, y: int):
        self.map[y][x] = GridItem.exit


class ReachExit(Task):
    def reward(self, state, action, state_prime) -> float:
        if state.x == state_prime.x and state.y == state_prime.y:
            return -1
            # return -5
        elif self.domain.map[state_prime.y][state_prime.x] == GridItem.exit:
            return 20
        else:
            return -1

    def stateisfinal(self, state) -> bool:
        item = self.domain.map[state.y][state.x]
        return item == GridItem.exit or item == GridItem.pit
