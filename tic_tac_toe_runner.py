from tic_tac_toe import RandomAgent, LearningAgent, WinTicTacToeTask, TicTacToeDomain

task = WinTicTacToeTask()
domain = TicTacToeDomain(3)

random_agent = RandomAgent("X", domain, task)
learning_agent = LearningAgent("O", domain, task)
agents = [random_agent, learning_agent]

winner = []

for i in range(0, 10000000):
    if i % 100 is 0:
        print(i)
    match_ended = False
    while not match_ended:
        for agent in agents:
            agent.see_result(domain.current_state())
            action = agent.choose_action(domain.current_state())
            domain.apply_action(action)
            # print(domain.current_state())
            if task.stateisfinal(domain.current_state()):
                match_ended = True
                winner.append(task.winner(domain.current_state()))
                for other_agent in agents:
                    if other_agent == agent:
                        pass
                    agent.see_result(domain.current_state())
                domain.reset()

print("Learning agent: " + str(winner.count(learning_agent.symbol)))
print("Random agent: " + str(winner.count(random_agent.symbol)))
print("Draw: " + str(winner.count(None)))

num_with_value = {}
for (key, value) in learning_agent.table.items():
    existing_value = num_with_value.get(value, 0.0)
    num_with_value[value] = int(existing_value + 1)

print(num_with_value)
