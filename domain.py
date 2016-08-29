from action import Action


class Domain():
    def apply_action(self, action: Action):
        raise NotImplementedError("Should have implemented this")

    def reset(self):
        raise NotImplementedError("Should have implemented this")
