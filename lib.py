'''
This file will mainly include action value related functions, different calculation
methods and related functions.
'''

import operator
import numpy as np


#  test set for functions evaluation those function way or not undergo some modification
#  depending on the input and output structure that we would evaluate as most suitable for 
#  large upcoming usage
num_iteration = 152
epsilon1 = 0.05 # I named it epsilon 1 cause epsilon is a token in python
action_value_state = {'a':1000, 'b':3000, 'c': 100, 'f': 321}
reward = {}

def greedy(action_value_dict):
    return max(action_value_dict.items(), key=operator.itemgetter(1))[0]

def epsilonGreedy(action_value_dict, epsy):
    action_type = np.random.choice(['exploitation', 'exploration'], p=[1-epsy, epsy])
    if action_type == 'exploitation':
        return max(action_value_dict.items(), key=operator.itemgetter(1))[0]
    else :
        expoitation_action = max(action_value_dict.items(), key=operator.itemgetter(1))[0]
        exploration_action_space = list(action_value_dict.keys())
        exploration_action_space.remove(expoitation_action)
        return np.random.choice(exploration_action_space)

def ActionValueUpdateComputation(action_value_dict, action, iteration, reward):
    old =action_value_dict[action]
    new_action_value = old + (1/iteration)*(reward-old)
    return new_action_value


print(greedy(action_value_state))
print(epsilonGreedy(action_value_state, epsilon1))
print(ActionValueUpdateComputation(action_value_state, 'c', 123, 3))