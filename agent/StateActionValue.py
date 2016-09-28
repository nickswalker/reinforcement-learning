from rl.action import Action
from rl.state import State


class StateActionValueTable:
    def __init__(self):
        self.table = []

    def reset(self):
        self.table = []

    def actionvalue(self, state: State, action: Action) -> float:
        # What if its not in the table?
        return self.table[state][action]

    def setactionvalue(self, state: State, action: Action, value: float):
        self.table[state][action] = value

    def bestactions(self, state) -> set[Action]:
        entry = self.table[state]

        best_actions = []
        best_value = float("-inf")
        for (action, value) in entry.iteritems():
            if value > best_value:
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return best_actions
