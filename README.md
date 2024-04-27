# Tiny DAG

Bare bones implementation of computation (directed, acyclic) graph for Python.

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

# Usage example

```
from tinydag.graph import Graph
from tinydag.node import Node

add = lambda a, b: a + b
mul = lambda a, b: a * b
div = lambda a, b: a / b

nodes = [
    Node(["add1", "x"], add, "add2"),
    Node(["add1", "add2"], mul, "mul"),
    Node(["x", "y"], add, "add1"),
    Node(["mul", "z"], div, "div"),
]
graph = Graph(nodes)
graph.render()
data = {"x": 5, "y": 3, "z": 3}
results = graph.calculate(data)
```

render method produces following figure:
<p align="center">
<img src="sample_graph.jpg" width="800px" />
</p>