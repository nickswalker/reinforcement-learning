from enum import Enum

from rl.action import Action
from rl.domain import Domain
from rl.state import State
from rl.task import Task


class Direction(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


class GridItem(Enum):
    empty = 0
    pit = 1
    exit = 2


class GridWorldState(State):
    def __init__(self, x: int, y: int, map: list[list[int]]):
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
        return str(self.x) + str(self.y)


class GridWorldAction(Action):
    def __init__(self, direction: Direction):
        super().__init__()
        self.direction = direction


class GridWorld(Domain):
    def __init__(self, width: int, height: int):
        self.map = [[0] * height for _ in range(0, width)]
        self.width = width
        self.height = height
        self.agent_x = 0
        self.agent_y = 0
        self.actions = [GridWorldAction(Direction.up), GridWorldAction(Direction.right),
                        GridWorldAction(Direction.down), GridWorldAction(Direction.left)]

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

    def current_state(self):
        return GridWorldState(self.agent_x, self.agent_y, self.map)

    def reset(self):
        self.map = [[None] * self.width for _ in range(0, self.height)]


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
