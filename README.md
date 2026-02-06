# 🧠 Micrograd Playground — Visual Autograd & Neural Network Explorer

An interactive **Streamlit-powered playground** for exploring **automatic differentiation**, **computational graphs**, and **neural network training** — built on top of a minimal Micrograd-style engine.

This project bridges the gap between **theory and intuition** by letting users *see* how gradients flow and how neural networks learn decision boundaries in real time.

---

## 🚀 Why This Project Matters

Modern deep learning frameworks hide a lot of magic.  
This app **peels back the abstraction** and visualizes what’s really happening under the hood:

- How expressions become **computational graphs**
- How **backpropagation** flows through those graphs
- How **MLPs learn nonlinear decision boundaries**
- How architecture, activation functions, and hyperparameters affect learning

This project is ideal for:
- Recruiters evaluating **ML fundamentals**
- Engineers learning **autograd internals**
- Students building intuition for **backpropagation**
- Anyone curious about how neural networks *actually* work

---

## 🧩 Features

### 1️⃣ Expression DAG Explorer
Interactively build and visualize computational graphs.

**What you can do:**
- Enter symbolic expressions using `a`, `b`, and `c`
- Apply activations like `relu`, `tanh`, `sigmoid`, `leaky_relu`
- Automatically compute gradients via backpropagation
- Visualize the full **directed acyclic graph (DAG)** using Graphviz

**Example Expression**

(a * b + c).tanh()

**Each node shows:**

Value, Gradient and Operation

This makes gradient flow explicit and inspectable.

### 2️⃣ Neural Network Training & Decision Boundaries

Train a fully-connected MLP from scratch and watch it learn.

**Supported datasets**

- XOR

- AND / OR

- Circle classification

- Two Moons

### Customizable

- Number of hidden layers (1–3)

- Neurons per layer

- Activation function

- Learning rate

- Training steps

**Live visualizations**

📉 Training loss curve

🎯 Decision boundary heatmap

🔴 Ground-truth data points

This is essentially a white-box neural network playground.

### 🙏 Acknowledgements

This project is inspired by Andrej Karpathy’s Micrograd, an educational automatic differentiation engine that demystifies backpropagation and neural networks.

Micrograd by Andrej Karpathy:
https://github.com/karpathy/micrograd
