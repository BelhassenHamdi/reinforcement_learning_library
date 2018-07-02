import operator



stats = {'a':1000, 'b':3000, 'c': 100, 'f': 321}

def greedy(action_value_dict):
    return max(action_value_dict.items(), key=operator.itemgetter(1))[0]

print(greedy(stats))