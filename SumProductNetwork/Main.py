from SumProductNetwork import SPN
from SumProductNetwork import opp_sum, opp_multi, variable

spn = SPN()

# Initialize variables
x1 = variable(name='x1', size=2)
x2 = variable(name='x2', size=2)
spn.addVar((x1, x2))

# The multiply operators
m1 = opp_multi(inherit=[(x1,0), (x2,0)])
m2 = opp_multi(inherit=[(x1,1), (x2,1)])
spn.addNodes((m1, m2))

# The final plus operator
p1 = opp_sum(inherit=[m1, m2], probs=[None, None])  # [0.3333333, 0.66666666]
spn.addNodes([p1])

#   +--------------+
#   |  x1  x2  p   |
#   |  0   0  0.3  |
#   |  0   1  0.0  |
#   |  1   0  0.0  |
#   |  1   1  0.7  |
#   +--------------+

data = []
for i in range(1):
    data.append((('x1',0), ('x2',0)))

for i in range(2):
    data.append((('x1',1), ('x2',1)))

spn.fit(data, method = "GD", num_epochs=20, alpha=0.1)

want = ['x1', 'x2']
given = []

print("The probabilities is:\n\n", spn.eval(want = want, given = given))
