import hashlib
import random
from copy import deepcopy
from typing import List

from action import Action
from agent import Agent
from domain import Domain
from state import State
from task import Task


class TicTacToeState(State):
    def __init__(self, representation: List[List[str]]):
        self.representation = representation

    def __hash__(self):
        return int(hashlib.md5(self.__str__()).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, TicTacToeState):
            return other.__hash__() == self.__hash__()
        return False

    def __str__(self):
        string_representation = str()
        for x in range(0, len(self.representation)):
            for y in range(0, len(self.representation[0])):
                if self.representation[y][x] is None:
                    string_representation += "_"
                else:
                    string_representation += self.representation[y][x]
            string_representation += "\n"
        return string_representation.encode()


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

    def prepare_for_new_episode(self):
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
        first = state.representation[goal - 1][goal - 1]

        for i in range(0, goal):
            element = state.representation[goal - 1 - i][goal - 1 - i]
            if first == element and element is not None:
                count += 1

        if count == goal:
            return first

        return None


class RandomAgent(TicTacToeAgent):
    def choose_action(self, state: State):
        options = []
        for x in range(0, len(state.representation)):
            for y in range(0, len(state.representation[0])):
                if state.representation[y][x] is None:
                    options.append((x, y))

        choice = random.choice(options)

        return TicTacToeAction(self.symbol, choice[0], choice[1])


class LearningAgent(TicTacToeAgent):
    def __init__(self, symbol: str, world, task):
        super().__init__(symbol, world, task)
        self.table = {}
        self.alpha = 0.9
        self.previous_state = None
        self.was_exploratory = False

    def value_of_state(self, state: State) -> float:
        existing_value = self.table.get(state)

        if existing_value is None:
            new_value = 0.5
            winner = self.task.winner(state)
            if winner == self.symbol:
                new_value = 1.0
            elif winner != self.symbol and winner is not None:
                new_value = 0.0
            self.table[state] = new_value

        return self.table.get(state)

    def value_of_action_in_state(self, state, x, y) -> float:
        representation_copy = deepcopy(state.representation)
        representation_copy[y][x] = self.symbol
        new_state = TicTacToeState(representation_copy)

        return self.value_of_state(new_state)

    def choose_action(self, state) -> Action:
        self.previous_state = state
        options = []
        for x in range(0, len(state.representation)):
            for y in range(0, len(state.representation[0])):
                if state.representation[y][x] is None:
                    options.append((x, y))

        if random.random() > 0.9:
            self.was_exploratory = True
            choice = random.choice(options)
            return TicTacToeAction(self.symbol, choice[0], choice[1])
        else:
            self.was_exploratory = False

        max_option = None
        max_value = float("-inf")
        for option in options:
            value = self.value_of_action_in_state(state, option[0], option[1])
            if value > max_value:
                max_value = value
                max_option = option

        assert max_option is not None
        return TicTacToeAction(self.symbol, max_option[0], max_option[1])

    def see_result(self, state_prime):
        if self.was_exploratory or self.previous_state is None:
            return
        prev_value = self.value_of_state(self.previous_state)
        prime_value = self.value_of_state(state_prime)

        error = prime_value - prev_value
        self.table[self.previous_state] = prev_value + self.alpha * error

    def prepare_for_new_episode(self):
        self.previous_state = None
