from micrograd.engine import AutoDiffNode
from micrograd.neural_network import MLP

a = AutoDiffNode(2.0)
b = AutoDiffNode(-3.0)
c = AutoDiffNode(10.0)

d = a * b + c
d.backward()

print("d=", d.data)
print("a.grad=", a.grad)
print("b.grad=", b.grad)
