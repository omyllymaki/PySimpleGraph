# Tiny DAG

A small library to orchestrate function calls using graph structure.

The library contains bare-bones implementation of computation (directed, acyclic) graph. User provides a graph structure
(nodes) and input data for the graph. The graph executes every node in the graph and returns output of every node as the 
result. The library supports multiple outputs per node, caching of the node results, and parallel execution of the nodes.

# Requirements

- Python >= 3.6
- graphviz (optional)

# Installation

Install graphviz (optional, needed for rendering)
```
sudo apt-get install graphviz
```

Install tiny-dag
```
pip3 install tiny-dag
```

# Usage

Here are the rules:
- Node functions need to return dict (or None) with keys matching node output definition.
- Output of the node is referenced in the graph structure by node_name/output_name.
- User needs to provide missing information, as dict, when calculate method is called.  

And thats it. Otherwise you are free to write any kind of functions and orchestrate calling of those functions by defining nodes that form the graph.

Usage example:
```
from tinydag.graph import Graph
from tinydag.node import Node

def add(a, b): return {"output": a + b}
def mul(a, b): return {"output": a * b}
def div(a, b): return {"output": a / b}
def add_subtract(a, b): return {"add_output": a + b, "subtract_output": a - b}

nodes = [
    Node(["add1/output", "x"], add, "add2", ["output"]),
    Node(["add1/output", "add2/output"], mul, "mul", ["output"]),
    Node(["x", "y"], add, "add1", ["output"]),
    Node(["x", "z"], add_subtract, "add_subtract", ["add_output", "subtract_output"]),
    Node(["mul/output", "add_subtract/add_output"], div, "div", ["output"]),
]

graph = Graph(nodes)
graph.render()

data = {"x": 5, "y": 3, "z": 3}
results = graph.calculate(data)
print(f"Result: {results}")
```

The results is dict of node outputs, in this case:

{'add1/output': 8, 
'add_subtract/add_output': 8, 
'add_subtract/subtract_output': 2, 
'add2/output': 13, 
'mul/output': 104, 
'div/output': 13.0}

render method produces following figure:
<p align="center">
<img src="sample_graph.jpg" width="800px" />
</p>

For a bit more complicated and practical usage, see how the library can be used to orchestrate and visualize data
processing pipelines: src/samples/sample_credit_risk_prediction.py.