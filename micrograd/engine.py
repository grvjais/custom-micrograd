class AutoDiffNode:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        # variables for internal use 
        self._backward = lambda: None     
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, AutoDiffNode) else AutoDiffNode(other)
        node = AutoDiffNode(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += node.grad
            other.grad += node.grad
        node._backward = _backward

        return node
    
    def __mul__(self, other):
        other = other if isinstance(other, AutoDiffNode) else AutoDiffNode(other)
        node = AutoDiffNode(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * node.grad
            other.grad += self.data * node.grad
        node._backward = _backward 

        return node

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "restricting to int and float powers only"
        node = AutoDiffNode(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other-1)) * node.grad
        node._backward = _backward

        return node  
    
    def relu(self):
        node = AutoDiffNode(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (node.data > 0) * node.grad

        node._backward = _backward

        return node
    
    def backward(self):

        order = []
        visited = set()
        def build_order(n):
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    build_order(child)
                order.append(n)

        build_order(self)

        # applying chain rule
        for n in reversed(order):
            n._backward()

    
    def __neg__(self):
            return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"AutoDiffNode(data={self.data}, grad={self.grad})"
        
