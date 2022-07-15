import numpy as np
import copy

RMAX = 20


def get_next_state(state, action):
    return state + (action - 1)


def check_convergence(action_values, action_values_temp):
    for row, row_temp in zip(action_values, action_values_temp):
        for action_value, action_value_temp in zip(row, row_temp):
            if (abs(action_value - action_value_temp) > 0.1):
                return True
    return False


def compute_values_alternative(exp):
    visits = range(10)
    states = range(10)
    actions = range(3)
    action_values = [[-1 for i in range(3)] for j in range(10)]
    rewards = [[-20, 20, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
               [-1, -1, -1], [-1, -1, -1], [-1, 20, -20]]
    discount_factor = 0.9
    min_visits = np.min(visits)
    not_converged = True
    action_values_temp = copy.deepcopy(action_values)
    while not_converged:
        for state in states:
            for action in actions:
                if (state == 0 and action == 0 or state == 9 and action == 2):
                    continue
                if exp and visits[state] == min_visits:
                    action_values_temp[state][action] = RMAX
                else:
                    reward = rewards[state][action]
                    best_next_action_reward = max(
                        action_values_temp[get_next_state(state, action)])
                    action_values_temp[state][action] = reward + discount_factor * best_next_action_reward
        not_converged = check_convergence(action_values, action_values_temp)
        print(action_values)
        action_values = copy.deepcopy(action_values_temp)
    return action_values


if __name__ == '__main__':
    action_values_comp = compute_values_alternative(False)
