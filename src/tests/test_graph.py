import unittest
from functools import partial

from tinydag.exceptions import InvalidGraphError, MissingInputError
from tinydag.graph import Graph
from tinydag.node import Node


def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def get_number(a):
    return a


class TestRaisesException(unittest.TestCase):

    def test_node_missing_input(self):
        nodes = [
            Node(["x", "y"], add, "add")
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1, "z": 2})

    def test_node_missing_input2(self):
        nodes = [
            Node(["x", "y"], add, "add")
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1})

    def test_node_missing_input3(self):
        nodes = [
            Node(["x", "y"], add, "add"),
            Node(["x", "z"], add, "add2")
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1, "y": 2})

    def test_not_unique_node_names(self):
        nodes = [
            Node(["x", "y"], add, "add"),
            Node(["add", "z"], mul, "add"),
        ]
        self.assertRaises(InvalidGraphError, Graph, nodes)

    def test_cyclic_graph(self):
        nodes = [
            Node(["x", "add4"], add, "add1"),
            Node(["add1", "z"], mul, "add2"),
            Node(["add1", "add2"], mul, "add3"),
            Node(["add3", "add2"], mul, "add4"),
        ]
        g = Graph(nodes)
        self.assertRaises(InvalidGraphError, g.calculate, {"x": 1, "y": 2, "z": 3})
        self.assertRaises(InvalidGraphError, g.check)


class TestOperations(unittest.TestCase):

    def test_add_and_mul(self):
        nodes = [
            Node(["x", "y"], add, "add"),
            Node(["add", "z"], mul, "mul"),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add"], 1 + 2)
        self.assertEqual(results["mul"], (1 + 2) * 2)

    def test_reverse_ordering_of_nodes(self):
        nodes = [
            Node(["add", "z"], mul, "mul"),
            Node(["x", "y"], add, "add"),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add"], 1 + 2)
        self.assertEqual(results["mul"], (1 + 2) * 2)

    def test_add_and_mul_additional_input(self):
        nodes = [
            Node(["x", "y"], add, "add"),
            Node(["add", "z"], mul, "mul"),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2, "w": 6}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add"], 1 + 2)
        self.assertEqual(results["mul"], (1 + 2) * 2)

    def test_no_input_for_one_node(self):
        get_5 = partial(get_number, a=5)
        nodes = [
            Node([], get_5, "5"),
            Node(["5", "x"], add, "add"),
        ]
        g = Graph(nodes)
        data = {"x": 1}
        results = g.calculate(data)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["5"], 5)
        self.assertEqual(results["add"], 1 + 5)

    def test_no_input_for_calculate(self):
        get_5 = partial(get_number, a=5)
        add_2 = partial(add, b=2)
        nodes = [
            Node([], get_5, "5"),
            Node(["5"], add_2, "add_2"),
        ]
        g = Graph(nodes)
        results = g.calculate()

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["5"], 5)
        self.assertEqual(results["add_2"], 5 + 2)

    def test_more_complex_graph(self):
        nodes = [
            Node(["add2", "z"], div, "div"),
            Node(["div", "add"], add, "add3"),
            Node(["x", "y"], add, "add"),
            Node(["add", "z"], mul, "mul"),
            Node(["add", "x"], add, "add2"),
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
        self.assertEqual(results["add"], add_expected)
        self.assertEqual(results["mul"], mul_expected)
        self.assertEqual(results["add2"], add2_expected)
        self.assertEqual(results["div"], div_expected)
        self.assertEqual(results["add3"], add3_expected)

    def test_multiple_non_connected_graphs(self):
        nodes = [
            Node(["add1", "x1"], add, "add2"),
            Node(["x1", "x2"], add, "add1"),
            Node(["x3", "x4"], mul, "mul"),
        ]
        graph = Graph(nodes)

        data = {"x1": 5, "x2": 3, "x3": 3, "x4": 2}
        results = graph.calculate(data)
        print(f"Result: {results}")

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add1"], 5 + 3)
        self.assertEqual(results["add2"], (5 + 3) + 5)
        self.assertEqual(results["mul"], 3 * 2)

    def test_add_one_node_to_existing_graph(self):
        nodes = [
            Node(["x", "y"], add, "add"),
        ]
        g = Graph(nodes)

        new_node = Node(["add", "z"], mul, "mul")
        g += new_node

        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), 2)
        self.assertEqual(results["add"], 1 + 2)
        self.assertEqual(results["mul"], (1 + 2) * 2)

    def test_add_multiple_nodes_to_existing_graph(self):
        nodes = [
            Node(["x", "y"], add, "add1"),
        ]
        g = Graph(nodes)

        new_nodes = [
            Node(["add1", "z"], mul, "mul"),
            Node(["add1", "z"], add, "add2"),
        ]
        g += new_nodes

        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data)

        self.assertEqual(len(results), 3)
        self.assertEqual(results["add1"], 1 + 2)
        self.assertEqual(results["mul"], (1 + 2) * 2)
        self.assertEqual(results["add2"], (1 + 2) + 2)


class TestRendering(unittest.TestCase):

    def test_and_and_mul(self):
        nodes = [
            Node(["x", "y"], add, "add"),
            Node(["add", "z"], mul, "mul"),
        ]
        g = Graph(nodes)
        g.render()


if __name__ == '__main__':
    unittest.main()
