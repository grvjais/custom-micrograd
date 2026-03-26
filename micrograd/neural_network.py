import random
from micrograd.engine import AutoDiffNode

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    

class Neuron(Module):

    def __init__(self, nin, act_fn = 'relu'):
        self.w = [AutoDiffNode(random.uniform(-1,1)) for weight in range(nin)]
        self.b = AutoDiffNode(0)
        self.act_fn = act_fn

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        
        if self.act_fn == 'relu':
            return act.relu()
        
        elif self.act_fn == 'tanh':
            return act.tanh()
        
        elif self.act_fn == 'sigmoid':
            return act.sigmoid()
        
        elif self.act_fn == 'linear':
            return act
        
        elif self.act_fn == 'leaky_relu':
            return act.leaky_relu()
        
        else:
            return act.relu()

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self.act_fn}Neuron({len(self.w)})"
        
class Layer(Module):
    
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n)for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, activation = 'relu'):
        size = [nin] + nouts
        self.layers = [Layer(size[i], 
                             size[i+1], 
                             act_fn = 'linear' if i == len(nouts)-1 else activation
                             ) 
                             for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"




