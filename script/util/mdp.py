import mdptoolbox.mdp as mdp


def value_iteration(T, R, gamma=0.99):
    vi = mdp.ValueIteration(T, R, gamma)
    vi.run()

    return vi

def q_learning(T, R, gamma=0.99):
    ql = mdp.QLearning(transitions=T, reward=R, discount=gamma)
    ql.setVerbose()
    ql.run()

    return ql
