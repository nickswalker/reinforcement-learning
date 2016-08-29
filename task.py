class Task(object):
    def __init__(self):
        pass

    def stateisfinal(self, state):
        raise NotImplementedError("Should have implemented this")

    def reward(self, state, action, state_prime):
        raise NotImplementedError("Should have implemented this")
