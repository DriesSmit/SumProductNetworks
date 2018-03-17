from SumProductNetwork import SPN
from SumProductNetwork import opp_sum, opp_multi, variable

spn = SPN()

# Initialize variables
x1 = variable(name='x1', size=2)
x2 = variable(name='x2', size=2)
spn.addVar((x1, x2))

# The plus operators
p1 = opp_sum(inherit=[(x1, 0), (x1, 1)], probs=[0.2, 0.8])
p2 = opp_sum(inherit=[(x1, 0), (x1, 1)], probs=[0.1, 0.9])
p3 = opp_sum(inherit=[(x2, 0), (x2, 1)], probs=[0.4, 0.6])
spn.addNodes((p1, p2, p3))

# The multiply operators
m1 = opp_multi(inherit=[p1, p3])
m2 = opp_multi(inherit=[p2, p3])
spn.addNodes((m1, m2))

# The final plus operator
p4 = opp_sum(inherit=[m1, m2], probs=[0.3, 0.7])
spn.addNodes([p4])

# -------------------
# |n_x1 x1 n_x2 x2|   p   |
# | 0   0   0   0 | 0.000 |
# | 0   0   0   1 | 0.000 |
# | 0   0   1   0 | 0.000 |
# | 0   0   1   1 | 0.000 |
# | 0   1   0   0 | 0.000 |
# | 0   1   0   1 | 0.522 |
# | 0   1   1   0 | 0.348 |
# | 0   1   1   1 | 0.870 |
# | 1   0   0   0 | 0.000 |
# | 1   0   0   1 | 0.078 |
# | 1   0   1   0 | 0.052 |
# | 1   0   1   1 | 0.130 |
# | 1   1   0   0 | 0.000 |
# | 1   1   0   1 | 0.600 |
# | 1   1   1   0 | 0.400 |
# | 1   1   1   1 | 1.000 |

want = ['x2']
given = []

print("The probabilities is:\n\n", spn.eval(want = want, given = given))
