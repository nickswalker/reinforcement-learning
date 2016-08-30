import random
from copy import deepcopy

from action import Action
from state import State
from tic_tac_toe import TicTacToeAgent, TicTacToeState, TicTacToeAction


class LearningAgent(TicTacToeAgent):
    def __init__(self, symbol: str, world, task):
        super().__init__(symbol, world, task)
        self.table = {}
        self.alpha = 0.2
        self.epsilon = 0.1
        self.previous_state = world.current_state()
        self.previous_move = None
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
            elif self.task.draw(state):
                new_value = 1.0

            self.table[state] = new_value

        return self.table.get(state)

    def value_of_action_in_state(self, state, x, y) -> float:
        return self.value_of_state(self.state_for_action(state, x, y))

    def state_for_action(self, state, x, y) -> State:
        representation_copy = deepcopy(state.representation)
        representation_copy[y][x] = self.symbol
        return TicTacToeState(representation_copy)

    def choose_action(self, state) -> Action:
        options = []
        for y in range(0, len(state.representation)):
            for x in range(0, len(state.representation[0])):
                if state.representation[y][x] is None:
                    options.append((x, y))

        if random.random() < self.epsilon:
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

        self.update_value(self.previous_state, state)
        self.previous_state = state

        self.previous_move = self.state_for_action(state, max_option[0], max_option[1])
        self.update_value(self.previous_move, state)

        return TicTacToeAction(self.symbol, max_option[0], max_option[1])

    def update_value(self, state, state_prime):
        if self.was_exploratory:
            return
        prev_value = self.value_of_state(state)
        prime_value = self.value_of_state(state_prime)

        error = prime_value - prev_value
        self.table[state] = prev_value + self.alpha * error

    def prepare_for_new_episode(self, state):
        self.previous_state = state

    def see_result(self, state: State):
        assert state != self.previous_state
        self.update_value(self.previous_state, state)
