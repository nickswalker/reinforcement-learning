class LinearVFA:
    def __init__(self, num_features, actions: List[Action], per_action_vfa=True, initial_value=0.0):
        self.num_features = num_features
        self.per_action_vfa = per_action_vfa
        self.actions = actions
        self.weights_per_action = []
        self.weights = None
        self.reset(initial_value)

    def reset(self, value=0.0):
        if self.per_action_vfa:
            for action in self.actions:
                self.weights_per_action[action] = [value for _ in range(0, self.num_features)]
        else:
            weights = [value for _ in range(0, self.num_features)]

    def actionvalue(self, features: List[float], action: Action) -> float:
        weights = None
        if self.per_action_vfa:
            weights = self.weights_per_action[]
        else:
            weights = self.weights

        return np.dot(weights, features)

    def statevalue(self, features: List[float]):
        raise Exception()

    def bestactions(self, features: List[float]) -> set[Action]:
        best_actions = []
        best_value = float("-inf")
        for action in self.actions:
            value = self.actionvalue(features, action)
            if value > best_value:
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return best_actions
