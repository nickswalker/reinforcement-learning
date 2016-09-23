import random

from rl.action import Action
from rl.agent import Agent
from rl.state import State


class RandomAgent(Agent):
    def choose_action(self, state: State) -> Action:
        available_actions = self.domain.get_actions(state)
        return random.sample(available_actions)
