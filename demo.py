import streamlit as st
import graphviz
import numpy as np
import matplotlib.pyplot as plt

from micrograd.engine import AutoDiffNode
from micrograd.neural_network import MLP

# ==========================================
# PART 2: GRAPHVIZ HELPER
# ==========================================

def draw_dot(root):
    dot = graphviz.Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    
    for n in nodes:
        uid = str(id(n))
        # Visual node for value/grad
        dot.node(name=uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label if n.label else '', n.data, n.grad), shape='record')
        if n._op:
            # Operation node
            dot.node(name=uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# ==========================================
# PART 3: STREAMLIT APP
# ==========================================

st.set_page_config(layout="wide", page_title="Micrograd Playground")
st.title("🧠 Micrograd Visualizer")

tab1, tab2 = st.tabs(["1. Expression DAG Explorer", "2. Neural Net Training & Boundary"])

# --- TAB 1: EXPRESSION PARSER ---
with tab1:
    st.header("Visualize Computational Graphs")
    st.markdown("Enter a mathematical expression using `a`, `b`, and `c`.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Inputs")
        a_val = st.slider("Value for a", -5.0, 5.0, 2.0)
        b_val = st.slider("Value for b", -5.0, 5.0, -3.0)
        c_val = st.slider("Value for c", -5.0, 5.0, 10.0)
        activation = st.selectbox(
            "Select Activation Function:",
            ["relu", "tanh", "sigmoid", "linear","leaky_relu"]
            )
        # User input expression
        expr = st.text_input("Expression", value=f"(a * b + c).{activation}()")
        st.caption("Supported: +, *, .tanh(), .exp(), .sigmoid(), etc.")

    with col2:
        try:
            # 1. Initialize Values
            a = AutoDiffNode(a_val, label='a')
            b = AutoDiffNode(b_val, label='b')
            c = AutoDiffNode(c_val, label='c')
            
            # 2. Safe Evaluate
            # We create a local scope so the user can only access specific variables
            safe_dict = {'a': a, 'b': b, 'c': c, 'Value': AutoDiffNode}
            
            # Eval allows us to parse the string "a*b" into actual code
            d = eval(expr, {"__builtins__": None}, safe_dict)
            d.label = 'out'
            
            # 3. Backward
            d.backward()
            
            # 4. Draw
            st.graphviz_chart(draw_dot(d))
            
            st.success(f"Output: {d.data:.4f}")
            
        except Exception as e:
            st.error(f"Invalid Expression: {e}")

# --- TAB 2: MLP VISUALIZER ---
with tab2:
    st.header("Interactive Neural Network Training")
    
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:

        st.subheader("Training Problem")
        problem = st.selectbox(
            "Select Problem Type:",
            ["XOR", "AND", "OR", "Circle", "Two Moons"]
        )

        st.subheader("Hyperparameters")
        lr = st.slider("Learning Rate", 0.01, 0.2, 0.1)
        steps = st.slider("Training Steps", 10, 500, 100)
        
        st.subheader("Network Architecture")
        n_hidden = st.slider("Hidden Neurons", 2, 20, 4)
        n_layer = st.slider("Hidden layers", 1, 3, 1)
        act = st.selectbox(
            "Activation Function:",
            ["relu", "tanh", "sigmoid", "linear","leaky_relu"]
            )
        
        def get_dataset(problem):

            if problem == "XOR":
                X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                y = [-1, 1, 1, -1]

            elif problem == "AND":
                X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                y = [-1, -1, -1, 1]

            elif problem == "OR":
                X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                y = [-1, 1, 1, 1]

            elif problem == "Circle":
                X = []
                y = []
                for x1 in np.linspace(-1.5, 1.5, 10):
                    for x2 in np.linspace(-1.5, 1.5, 10):
                        X.append([x1, x2])
                        y.append(1 if x1**2 + x2**2 < 1 else -1)

            elif problem == "Two Moons":
                X, y = [], []
                for t in np.linspace(0, np.pi, 50):
                    X.append([np.cos(t), np.sin(t)])
                    y.append(1)
                    X.append([1 - np.cos(t), 1 - np.sin(t)])
                    y.append(-1)

            return X, y


        if st.button("Train Model"):

            X, y = get_dataset(problem)

            # Saving it for visualization later
            st.session_state['X'] = X
            st.session_state['y'] = y

            # Initialize Model
            if n_layer == 1:
                model = MLP(2, [n_hidden, 1], activation=act)
                
            elif n_layer == 2:
                model = MLP(2, [n_hidden, n_hidden, 1], activation=act)

            elif n_layer == 3:
                model = MLP(2, [n_hidden, n_hidden, n_hidden, 1], activation=act)

            
            # Training Loop
            loss_history = []
            progress_bar = st.progress(0)
            
            for k in range(steps):
                # forward
                total_loss = 0
                for xi, yi in zip(X, y):
                    score = model(xi)
                    loss = (score - yi)**2 
                    total_loss += loss.data
                    
                    # backward
                    model.zero_grad()
                    loss.backward()
                    
                    # update
                    for p in model.parameters():
                        p.data -= lr * p.grad
                
                loss_history.append(total_loss)
                progress_bar.progress((k+1)/steps)
            
            st.success(f"Final Loss: {total_loss:.4f}")
            st.session_state['trained_model'] = model
            st.session_state['loss_history'] = loss_history

    with col_viz:
        if 'trained_model' in st.session_state:
            # Plot 1: Loss Curve
            st.subheader("Training Loss")
            st.line_chart(st.session_state['loss_history'])
            
            # Plot 2: Decision Boundary
            st.subheader("Decision Boundary")
            
            model = st.session_state['trained_model']
            
            # Create a meshgrid
            h = 0.25
            x_min, x_max = -2, 2
            y_min, y_max = -2, 2
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            # Predict for every point in meshgrid
            Z = []
            for inputs in zip(xx.ravel(), yy.ravel()):
                val = model(list(inputs)).data
                Z.append(val)
                
            Z = np.array(Z).reshape(xx.shape)
            
            fig, ax = plt.subplots()
            # Contour plot
            c = ax.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.8)
            plt.colorbar(c)
            
            # Plot original data points
            X_data = np.array(st.session_state['X'])
            y_data = np.array(st.session_state['y'])
            ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, s=100, cmap='coolwarm', edgecolors='k')
            ax.set_title("Neural Net 'Thought' Process")
            
            st.pyplot(fig)
        else:
            st.info("Click 'Train Model' to see the visualization.")
