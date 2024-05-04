import logging

from tinydag.graph import Graph
from tinydag.node import Node

logging.basicConfig(level=logging.INFO)


def add(a, b):
    return {"output": a + b}


def mul(a, b):
    return {"output": a * b}


def div(a, b):
    return {"output": a / b}


def add_subtract(a, b):
    return {"add_output": a + b, "subtract_output": a - b}


def main():
    nodes = [
        Node(["add1/output", "x"], add, "add2", ["output"]),
        Node(["add1/output", "add2/output"], mul, "mul", ["output"]),
        Node(["x", "y"], add, "add1", ["output"]),
        Node(["x", "z"], add_subtract, "add_subtract", ["add_output", "subtract_output"]),
        Node(["mul/output", "add_subtract/add_output"], div, "div", ["output"]),
    ]

    graph = Graph(nodes)
    print("Graph: ", graph)
    graph.render()

    data = {"x": 5, "y": 3, "z": 3}
    results = graph.calculate(data, parallel=False)
    print(f"Result: {results}")


if __name__ == "__main__":
    main()
