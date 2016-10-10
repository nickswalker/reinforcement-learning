import random

from agent.StateActionValue import StateActionValueTable
from rl.action import Action
from rl.agent import Agent
from rl.domain import Domain
from rl.task import Task


class SarsaAgent(Agent):
    def __init__(self, domain: Domain, task: Task, epsilon=0.05, alpha=0.3, gamma=0.5, lamb=0.4):
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
        self.previousaction = None
        self.previousstate = None

        self.state_action_value_table = StateActionValueTable(domain.get_actions(domain.get_current_state()))
        self.current_cumulative_reward = 0.0

    def act(self):
        """Execute one action on the world, possibly terminating the episode.

        """
        state = self.domain.get_current_state()
        action = self.chooseaction(state)

        self.world.apply_action(action)

        state_prime = self.world.get_current_state()

        # For the first time step, we won't have received a reward yet.
        # We're just notifying the learner of our starting state and action.
        if self.previousstate is None and self.previousaction is None:
            ()
        else:
            reward = self.task.reward(state, action, state_prime)
            old_value = self.state_action_value_table.actionvalue(state, action)

            expectation = self.expected_value(state_prime)

            error = reward + self.gamma * expectation - old_value
            new_value = old_value + self.alpha * error
            self.state_action_value_table.setactionvalue(state, action, new_value)

        self.previousaction = action
        self.previousstate = state

        if self.task.stateisfinal(state_prime):
            reward = self.task.reward(self.previousstate, self.previousaction, state_prime)
            self.current_cumulative_reward += reward
            # Reset episode related information
            self.previousaction = None
            self.previousstate = None

    def chooseaction(self, state) -> Action:
        """Given a state, pick an action according to an epsilon-greedy policy.

        :param state: The state from which to act.
        :return:
        """
        if random.random() < self.epsilon:
            actions = self.domain.get_actions(state)
            return random.sample(actions)
        else:
            best_actions = self.state_action_value_table.bestactions(state)
            return random.sample(best_actions, 1)[0]

    def expected_value(self, state):
        actions = self.domain.get_actions(state)
        expectation = 0.0
        best_actions = self.state_action_value_table.bestactions(state)
        num_best_actions = len(best_actions)
        nonoptimal_mass = self.epsilon

        if num_best_actions > 0:
            a_best_action = random.sample(best_actions, 1)[0]
            greedy_mass = (1.0 - self.epsilon)
            expectation += greedy_mass * self.state_action_value_table.actionvalue(state, a_best_action)
        else:
            nonoptimal_mass = 1.0

        nonoptimal_actions = actions.difference(best_actions)

        # No best action, equiprobable random policy
        total_value = 0.0
        for action in nonoptimal_actions:
            total_value += self.state_action_value_table.actionvalue(state, action)
        expectation = nonoptimal_mass * total_value / len(actions)

        return expectation

    def get_cumulative_reward(self):
        return self.current_cumulative_reward

    def episode_ended(self, state):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None

    def logepisode(self):
        print("Episode reward: " + str(self.current_cumulative_reward))
