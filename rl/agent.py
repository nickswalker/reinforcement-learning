from rl.action import Action
from rl.domain import Domain
from rl.task import Task


class Agent(object):
    """
    An agent takes actions from the action space and applies them to the
    world. Its knowledge comes from the states returned by the world.
    """

    def __init__(self, domain: Domain, task: Task):
        self.domain = domain
        self.task = task

    def act(self):
        """
        Execute one step in the environment, possible terminating an episode
        :return:
        """
        raise NotImplementedError("Should have implemented this")

    def choose_action(self, state) -> Action:
        raise NotImplementedError("Should have implemented this")

    def episode_ended(self, state):
        """
        A way to let the agent observe the final transition in episodic tasks.
        :param state:
        :return:
        """
        raise NotImplementedError("Should have implemented this")

    def get_cumulative_reward(self) -> float:
        """
        Gets the cumulative reward for the current episode, or for the agent's life if non-episodic tasks
        :return:
        """
        raise NotImplementedError("Should have implemented this")
