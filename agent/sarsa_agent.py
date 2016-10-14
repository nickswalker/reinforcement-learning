import random

from agent.state_action_value_table import StateActionValueTable
from gridworld import GridWorldState, GridWorldAction, Direction, Set
from rl.action import Action
from rl.agent import Agent
from rl.domain import Domain
from rl.state import State
from rl.task import Task


class SarsaAgent(Agent):
    def __init__(self, domain: Domain, task: Task, epsilon=0.1, alpha=0.6, gamma=0.95, lamb=0.95, expected=False):
        """
        :param domain: The world the agent is placed in.
        :param task: The task in the world, which defines the reward function.
        """
        super().__init__(domain, task)
        self.world = domain
        self.task = task
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.expected = expected
        self.previousaction = None
        self.previousstate = None

        self.value_function = StateActionValueTable(domain.get_actions(domain.get_current_state()))
        self.current_cumulative_reward = 0.0

    def act(self):
        """Execute one action on the world, possibly terminating the episode.

        """
        state = self.domain.get_current_state()
        action = self.choose_action(state)

        self.world.apply_action(action)

        # For the first time step, we won't have received a reward yet.
        # We're just notifying the learner of our starting state and action.
        if self.previousstate is None and self.previousaction is None:
            ()
        else:
            self.update(self.previousstate, self.previousaction, state, action)

        self.previousaction = action
        self.previousstate = state

        state_prime = self.world.get_current_state()
        if self.task.stateisfinal(state_prime):
            self.update(state, action, state_prime, None, terminal=True)
            # Reset some episode related information
            self.previousaction = None
            self.previousstate = None

    def update(self, state: State, action: Action, state_prime: State, action_prime: Action, terminal=False):
        reward = self.task.reward(state, action, state_prime)
        old_value = self.value_function.actionvalue(state, action)

        # Terminal states are defined to have value 0
        if terminal:
            target = 0
        else:
            if self.expected:
                target = self.expected_value(state_prime)

            else:
                target = self.value_function.actionvalue(state_prime, action_prime)

        error = reward + self.gamma * target - old_value
        new_value = old_value + self.alpha * error
        self.value_function.setactionvalue(state, action, new_value)
        self.current_cumulative_reward += reward

    def choose_action(self, state) -> Action:
        """Given a state, pick an action according to an epsilon-greedy policy.

        :param state: The state from which to act.
        :return:
        """
        if random.random() < self.epsilon:
            actions = self.domain.get_actions(state)
            return random.sample(actions, 1)[0]
        else:
            best_actions = self.value_function.bestactions(state)
            return random.sample(best_actions, 1)[0]

    def expected_value(self, state):
        actions = self.domain.get_actions(state)
        expectation = 0.0
        best_actions = self.value_function.bestactions(state)
        num_best_actions = len(best_actions)
        nonoptimal_mass = self.epsilon

        if num_best_actions > 0:
            a_best_action = random.sample(best_actions, 1)[0]
            greedy_mass = (1.0 - self.epsilon)
            expectation += greedy_mass * self.value_function.actionvalue(state, a_best_action)
        else:
            nonoptimal_mass = 1.0

        if nonoptimal_mass > 0.0:
            # No best action, equiprobable random policy
            total_value = 0.0
            for action in actions:
                total_value += self.value_function.actionvalue(state, action)
            expectation += nonoptimal_mass * total_value / len(actions)

        return expectation

    def get_cumulative_reward(self):
        return self.current_cumulative_reward

    def episode_ended(self):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None

    def logepisode(self):
        print("Episode reward: " + str(self.current_cumulative_reward))

    def _log_table(self) -> str:
        result = ""
        action = GridWorldAction(Direction.up)
        for x in range(0, len(self.domain.map[0])):
            for y in range(0, len(self.domain.map)):
                test_state = GridWorldState(x, y, self.domain.map)
                result += "%.2f " % self.value_function.actionvalue(test_state, action)
            result += "\n"
        return result

    def _log_policy(self) -> str:
        result = "___________\n"
        for y in range(0, len(self.domain.map)):
            result += "|"
            for x in range(0, len(self.domain.map[0])):
                test_state = GridWorldState(x, y, self.domain.map)
                actions = self.value_function.bestactions(test_state)
                result += " " + self._best_actions_to_str(actions)
            result += "|\n"
        result += "------------"
        return result

    def _best_actions_to_str(self, actions: Set[Action]) -> str:
        actions = list(actions)
        if len(actions) == 1:
            action = actions[0]
            if action.direction == Direction.up:
                return "↑"
            if action.direction == Direction.right:
                return "→"
            if action.direction == Direction.down:
                return "↓"
            if action.direction == Direction.left:
                return "←"
        elif len(actions) == 4:
            return "*"
        else:
            return "."
