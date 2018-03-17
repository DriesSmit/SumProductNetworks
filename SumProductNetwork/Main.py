from SumProductNetwork import SPN
from SumProductNetwork import opp_sum, opp_multi, variable

spn = SPN()

# Initialize variables
x1 = variable(name='x1', size=2)
x2 = variable(name='x2', size=2)
x3 = variable(name='x3', size=2)
spn.addVar((x1, x2, x3))

# The multiply operators
m1 = opp_multi(inherit=[(x2, 0), (x3, 0)])
m2 = opp_multi(inherit=[(x2, 1), (x3, 1)])
m3 = opp_multi(inherit=[(x2, 0), (x3, 1)])
m4 = opp_multi(inherit=[(x2, 1), (x3, 0)])
spn.addNodes((m1, m2, m3, m4))

# The plus operators
p1 = opp_sum(inherit=[m1, m2], probs=[0.5, 0.5])
p2 = opp_sum(inherit=[m3, m4], probs=[0.5, 0.5])
spn.addNodes((p1, p2))

# The multiply operators
m5 = opp_multi(inherit=[(x1, 0), p1])
m6 = opp_multi(inherit=[p2, (x1, 1)])
spn.addNodes((m5, m6))

# The final plus operator
p3 = opp_sum(inherit=[m5, m6], probs=[0.5, 0.5])
spn.addNodes([p3])

want = ['x3']
given = [('x1',0), ('x2',1)]

print("The probabilities is:\n\n", spn.eval(want = want, given = given))
