import mdptoolbox.mdp as mdp


def value_iteration(T, R, gamma=0.99):
    vi = mdp.ValueIteration(T, R, gamma)
    vi.run()

    return vi

def q_learning(T, R, gamma=0.99):
    ql = mdp.QLearning(T, R, gamma)
    ql.run()

    return ql
