import hashlib
import random
from copy import deepcopy
from typing import List

from rl.action import Action
from rl.agent import Agent
from rl.domain import Domain
from rl.state import State
from rl.task import Task


class TicTacToeState(State):
    def __init__(self, representation: List[List[str]]):
        self.representation = representation

    def __hash__(self):
        return int(hashlib.md5(self.__str__().encode()).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, TicTacToeState):
            return other.__hash__() == self.__hash__()
        return False

    def __str__(self):
        string_representation = str()
        for y in range(0, len(self.representation)):
            for x in range(0, len(self.representation[0])):
                if self.representation[y][x] is None:
                    string_representation += "_"
                else:
                    string_representation += self.representation[y][x]
            string_representation += "\n"
        return string_representation


class TicTacToeAction(Action):
    def __init__(self, symbol: str, x: int, y: int):
        super().__init__()
        self.symbol = symbol
        self.x = x
        self.y = y


class TicTacToeAgent(Agent):
    def __init__(self, symbol: str, domain: Domain, task: Task):
        super().__init__(domain, task)
        self.symbol = symbol

    def see_result(self, state: State):
        return

    def prepare_for_new_episode(self, state):
        return


class TicTacToeDomain(Domain):
    def __init__(self, size: int):
        self.representation = [[None] * size for _ in range(0, size)]
        self.size = size

    def apply_action(self, action: Action):
        assert isinstance(action, TicTacToeAction)
        assert self.representation[action.y][action.x] is None
        self.representation[action.y][action.x] = action.symbol

    def current_state(self):
        return TicTacToeState(deepcopy(self.representation))

    def reset(self):
        self.representation = [[None] * self.size for _ in range(0, self.size)]


class WinTicTacToeTask(Task):
    def reward(self, state, action, state_prime) -> float:
        pass

    def stateisfinal(self, state) -> bool:
        if self.winner(state) is not None or self.draw(state):
            return True
        return False

    def draw(self, state) -> bool:
        for row in state.representation:
            for item in row:
                if item is None:
                    return False

        return True

    def winner(self, state) -> str:
        goal = len(state.representation)

        for row in state.representation:
            first = row[0]
            count = 0
            for element in row:
                if element != first or element is None:
                    break
                else:
                    count += 1

            if count == goal:
                return first

        for col in range(0, goal):
            first = state.representation[0][col]
            count = 0
            for row in range(0, goal):
                element = state.representation[row][col]
                if element != first or element is None:
                    break
                else:
                    count += 1

                if count == goal:
                    return first

        count = 0
        first = state.representation[0][0]
        for i in range(0, goal):
            element = state.representation[i][i]
            if first == element and element is not None:
                count += 1

        if count == goal:
            return first

        count = 0
        first = state.representation[goal - 1][0]

        for i in range(0, goal):
            element = state.representation[goal - 1 - i][i]
            if first == element and element is not None:
                count += 1

        if count == goal:
            return first

        return None


class RandomAgent(TicTacToeAgent):
    def choose_action(self, state: State):
        options = []
        for y in range(0, len(state.representation)):
            for x in range(0, len(state.representation[0])):
                if state.representation[y][x] is None:
                    options.append((x, y))

        choice = random.choice(options)

        return TicTacToeAction(self.symbol, choice[0], choice[1])


class InteractiveAgent(TicTacToeAgent):
    def choose_action(self, state) -> Action:
        chosen_x = int(input("x"))
        chosen_y = int(input("y"))

        return TicTacToeAction(self.symbol, chosen_x, chosen_y)
