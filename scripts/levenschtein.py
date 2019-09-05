from collections import Counter
from time import time

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        key = str(args) + str(kwargs)
        helper.c[key] += 1
        return func(*args, **kwargs)
    helper.c = Counter()
    helper.calls = 0
    helper.__name__= func.__name__
    return helper

# Recursive solution
@call_counter
def recursive_LD(word_1, word_2):
    if len(word_1) == 0:
        return len(word_2)
    if len(word_2) == 0:
        return len(word_1)
    if word_1[-1] == word_2[-1]:
        cost = 0
    else:
        cost = 1

    distance = min([recursive_LD(word_1[:-1], word_2) + 1,
                    recursive_LD(word_1, word_2[:-1]) + 1,
                    recursive_LD(word_1[:-1], word_2[:-1]) + cost])

    return distance


memo = {}
@call_counter
def recursive_LD_memo(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    cost = 0 if s[-1] == t[-1] else 1

    i1 = (s[:-1], t)
    if not i1 in memo:
        memo[i1] = recursive_LD_memo(*i1)
    i2 = (s, t[:-1])
    if not i2 in memo:
        memo[i2] = recursive_LD_memo(*i2)
    i3 = (s[:-1], t[:-1])
    if not i3 in memo:
        memo[i3] = recursive_LD_memo(*i3)
    res = min([memo[i1] + 1, memo[i2] + 1, memo[i3] + cost])

    return res

# Iterative solution
@call_counter
def iterative_LD(word_1, word_2):
    rows = len(word_2) + 1
    cols = len(word_1) + 1

    D = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(cols):
        D[0][i] = i
    for j in range(rows):
        D[j][0] = j

    for row in range(1, rows):
        for col in range(1, cols):
            if word_1[col - 1] == word_2[row - 1]:
                cost = 0
            else:
                cost = 1

            D[row][col] = min([D[row - 1][col] + 1,
                               D[row][col - 1] + 1,
                               D[row - 1][col - 1] + cost])

    return D[-1][-1]

start = time()
print(iterative_LD('chirag', 'angelica'))
end = time()
iterative_LD_time = end-start
print("iterative_LD was called {} times".format(str(iterative_LD.calls)))
print(iterative_LD.c)
print("iterative_LD took: {} seconds".format(iterative_LD_time))

start = time()
print(recursive_LD('chirag', 'angelica'))
end = time()
recursive_LD_time = end-start
print("recursive_LD was called {} times".format(str(recursive_LD.calls)))
print(recursive_LD.c)
print("recursive_LD took: {} seconds".format(recursive_LD_time))


print('Iterative LD was {} times quicker than recursive LD'.format((recursive_LD_time)/iterative_LD_time))

start = time()
print(recursive_LD_memo('chirag', 'angelica'))
end = time()
recursive_LD_memo_time = end-start
print("recursive_LD_memo was called {} times".format(str(recursive_LD_memo.calls)))
print(recursive_LD_memo.c)
print("recursive_LD_memo took: {} seconds".format(recursive_LD_memo_time))