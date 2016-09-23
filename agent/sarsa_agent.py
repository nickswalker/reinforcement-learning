import random

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
        super.__init__(domain, task)
        self.world = domain
        self.task = task
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.previousaction = None
        self.previousstate = None

        self.state_action_value_table = []
        self.current_cumulative_reward = 0.0

    def act(self, maximize=None):
        """Execute one action on the world, possibly terminating the episode.

        """
        state = self.domain.get_current_state()
        action = self.chooseaction(state)

        self.world.apply_action(action)

        state_prime = self.world.get_current_state()
        # For the first time step, we won't have received a reward yet.
        # We're just notifying the learner of our starting state and action.
        if self.previousstate is None and self.previousaction is None:

        else:
            reward = self.task.reward(state, action, state_prime)
            old_value = self.state_action_value_table[state][action]

        self.previousaction = action
        self.previousstate = state

        if self.task.stateisfinal(state_prime):
            reward = self.task.reward(self.previousstate, self.previousaction, state_prime)
            self.learner.end(reward)
            self.episode_reward += reward
            # Reset episode related information
            self.previousaction = None
            self.previousstate = None

    def chooseaction(self, state):
        """Given a state, pick an action according to an epsilon-greedy policy.

        :param state: The state from which to act.
        :return:
        """
        if random.random() < self.epsilon:
            actions = self.domain.get_actions(state)
            return random.sample(actions)

        return Action(optimal_params[0], optimal_params[1])

    def _learn(self, state_prime, action_prime):
        """Wraps the learning method of TD-lambda.

        :param state_prime:
        :param action_prime:
        """
        state = self.previousstate
        action = self.previousaction
        reward = self.task.reward(state, action, state_prime)

        # We handle terminal rewards separately, so the reward here should
        # always be negative.
        assert reward < 0

        # The learner is smart; it keeps copies of the previous states and
        # actions. We don't need to pass them in.
        self.learner.step(reward, self._compose(state_prime, action_prime))
        self.episode_reward += reward

    def get_cumulative_reward(self):
        return self.current_cumulative_reward

    def episode_ended(self, state):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None

    def logepisode(self):
        print("Episode reward: " + str(self.current_cumulative_reward))
