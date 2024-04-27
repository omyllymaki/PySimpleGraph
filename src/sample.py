import logging

from tinydag.graph import Graph
from tinydag.node import Node

logging.basicConfig(level=logging.DEBUG)


def main():
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
    print("Graph: ", graph)
    graph.render()

    data = {"x": 5, "y": 3, "z": 3}
    graph.check(data)
    results = graph.calculate(data)
    print(f"Result: {results}")


if __name__ == "__main__":
    main()
