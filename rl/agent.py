from action import Action
from domain import Domain
from task import Task


class Agent(object):
    """
    An agent takes actions from the action space and applies them to the
    world. Its knowledge comes from the states returned by the world.
    """

    def __init__(self, domain: Domain, task: Task):
        self.domain = domain
        self.task = task

    def choose_action(self, state) -> Action:
        raise NotImplementedError("Should have implemented this")
