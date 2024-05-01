import unittest
from functools import partial

from tinydag.exceptions import InvalidGraphError, MissingInputError
from tinydag.graph import Graph
from tinydag.node import Node


def add(a, b):
    return {"output": a + b}


def mul(a, b):
    return {"output": a * b}


def div(a, b):
    return {"output": a / b}


def get_number(a):
    return {"output": a}


def add_subtract(a, b):
    return {"add_output": a + b, "subtract_output": a - b}


class TestRaisesException(unittest.TestCase):

    def test_node_missing_input(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1, "z": 2})

    def test_node_missing_input2(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1})

    def test_node_missing_input3(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["x", "z"], add, "add2", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1, "y": 2})

    def test_not_unique_node_names(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add", "z"], mul, "add", ["output"]),
        ]
        self.assertRaises(InvalidGraphError, Graph, nodes)

    def test_cyclic_graph(self):
        nodes = [
            Node(["x", "add4/output"], add, "add1", ["output"]),
            Node(["add1/output", "z"], mul, "add2", ["output"]),
            Node(["add1/output", "add2/output"], mul, "add3", ["output"]),
            Node(["add3/output", "add2/output"], mul, "add4", ["output"]),
        ]
        g = Graph(nodes)
        self.assertRaises(InvalidGraphError, g.calculate, {"x": 1, "y": 2, "z": 3})
        self.assertRaises(InvalidGraphError, g.check)


class TestOperations(unittest.TestCase):

    def test_add_and_mul(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add/output", "z"], mul, "mul", ["output"]),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add/output"], 1 + 2)
        self.assertEqual(results["mul/output"], (1 + 2) * 2)

    def test_reverse_ordering_of_nodes(self):
        nodes = [
            Node(["add/output", "z"], mul, "mul", ["output"]),
            Node(["x", "y"], add, "add", ["output"]),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add/output"], 1 + 2)
        self.assertEqual(results["mul/output"], (1 + 2) * 2)

    def test_add_and_mul_additional_input(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add/output", "z"], mul, "mul", ["output"]),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2, "w": 6}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add/output"], 1 + 2)
        self.assertEqual(results["mul/output"], (1 + 2) * 2)

    def test_no_input_for_one_node(self):
        get_5 = partial(get_number, a=5)
        nodes = [
            Node([], get_5, "5", ["output"]),
            Node(["5/output", "x"], add, "add", ["output"]),
        ]
        g = Graph(nodes)
        data = {"x": 1}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["5/output"], 5)
        self.assertEqual(results["add/output"], 1 + 5)

    def test_no_input_for_calculate(self):
        get_5 = partial(get_number, a=5)
        add_2 = partial(add, b=2)
        nodes = [
            Node([], get_5, "5", ["output"]),
            Node(["5/output"], add_2, "add_2", ["output"]),
        ]
        g = Graph(nodes)
        results = g.calculate()

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["5/output"], 5)
        self.assertEqual(results["add_2/output"], 5 + 2)

    def test_more_complex_graph(self):
        nodes = [
            Node(["add2/output", "z"], div, "div", ["output"]),
            Node(["div/output", "add/output"], add, "add3", ["output"]),
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add/output", "z"], mul, "mul", ["output"]),
            Node(["add/output", "x"], add, "add2", ["output"]),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        add_expected = 1 + 2
        mul_expected = add_expected * 2
        add2_expected = add_expected + 1
        div_expected = add2_expected / 2
        add3_expected = add_expected + div_expected

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add/output"], add_expected)
        self.assertEqual(results["mul/output"], mul_expected)
        self.assertEqual(results["add2/output"], add2_expected)
        self.assertEqual(results["div/output"], div_expected)
        self.assertEqual(results["add3/output"], add3_expected)

    def test_multiple_non_connected_graphs(self):
        nodes = [
            Node(["add1/output", "x1"], add, "add2", ["output"]),
            Node(["x1", "x2"], add, "add1", ["output"]),
            Node(["x3", "x4"], mul, "mul", ["output"]),
        ]
        graph = Graph(nodes)

        data = {"x1": 5, "x2": 3, "x3": 3, "x4": 2}
        results = graph.calculate(data)
        print(f"Result: {results}")

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add1/output"], 5 + 3)
        self.assertEqual(results["add2/output"], (5 + 3) + 5)
        self.assertEqual(results["mul/output"], 3 * 2)

    def test_add_one_node_to_existing_graph(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
        ]
        g = Graph(nodes)

        new_node = Node(["add/output", "z"], mul, "mul", ["output"])
        g += new_node

        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), 2)
        self.assertEqual(results["add/output"], 1 + 2)
        self.assertEqual(results["mul/output"], (1 + 2) * 2)

    def test_add_multiple_nodes_to_existing_graph(self):
        nodes = [
            Node(["x", "y"], add, "add1", ["output"]),
        ]
        g = Graph(nodes)

        new_nodes = [
            Node(["add1/output", "z"], mul, "mul", ["output"]),
            Node(["add1/output", "z"], add, "add2", ["output"]),
        ]
        g += new_nodes

        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), 3)
        self.assertEqual(results["add1/output"], 1 + 2)
        self.assertEqual(results["mul/output"], (1 + 2) * 2)
        self.assertEqual(results["add2/output"], (1 + 2) + 2)


class TestRendering(unittest.TestCase):

    def test_and_and_mul(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add/output", "z"], mul, "mul", ["output"]),
        ]
        g = Graph(nodes)
        g.render()


if __name__ == '__main__':
    unittest.main()
