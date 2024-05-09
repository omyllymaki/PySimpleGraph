import logging
import os
import shutil
import unittest
from functools import partial
from os.path import exists

from tinydag.exceptions import InvalidGraphError, MissingInputError, InvalidNodeFunctionOutput
from tinydag.graph import Graph
from tinydag.node import Node

logging.basicConfig(level=logging.INFO)


def add(a, b):
    return {"output": a + b}


def mul(a, b):
    return {"output": a * b}


def div(a, b):
    return {"output": a / b}


def get_number(a):
    return {"output": a}


def add_return_none(a, b):
    c = a + b
    return


def zero_division(a):
    return {"output": a / 0}


def invalid_add_func(a, b):
    return a + b  # Should return dict


def add_subtract(a, b):
    return {"add_output": a + b, "subtract_output": a - b}


def add_lists(a, b):
    return {"output": [i + j for i, j in zip(a, b)]}


def mul_lists(a, b):
    return {"output": [i * j for i, j in zip(a, b)]}


class BaseTest(unittest.TestCase):

    def setUp(self):
        self.run_parallel = self._get_parallel_arg()

    @staticmethod
    def _get_parallel_arg():
        parallel_str = os.getenv('PARALLEL', 'False')
        return parallel_str.lower() == 'true'


class TestRaisesException(BaseTest):

    def test_node_missing_input(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1, "z": 2}, parallel=self.run_parallel)

    def test_node_missing_input2(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1}, parallel=self.run_parallel)

    def test_node_missing_input3(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["x", "z"], add, "add2", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1, "y": 2}, parallel=self.run_parallel)

    def test_missing_input4(self):
        nodes = [
            Node(["x", "y"], add, "add"),
            Node(["add/output", "x"], mul, "mul", ["output"]),
        ]
        g = Graph(nodes)
        self.assertRaises(MissingInputError, g.calculate, {"x": 1, "y": 2}, parallel=self.run_parallel)

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
        self.assertRaises(InvalidGraphError, g.calculate, {"x": 1, "y": 2, "z": 3}, parallel=self.run_parallel)
        self.assertRaises(InvalidGraphError, g.validate_graph)

    def test_function_doesnt_return_dict(self):
        nodes = [
            Node(["x", "y"], invalid_add_func, "add", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(InvalidNodeFunctionOutput, g.calculate, {"x": 1, "y": 2}, parallel=self.run_parallel)

    def test_function_doesnt_return_all_the_required_outputs(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output", "output2"])
        ]
        g = Graph(nodes)
        self.assertRaises(InvalidNodeFunctionOutput, g.calculate, {"x": 1, "y": 2}, parallel=self.run_parallel)

    def test_reading_from_non_existing_cache(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add/output", "z"], mul, "mul", ["output"]),
        ]

        cache_dir = "test_cache_dir"
        if exists(cache_dir):
            shutil.rmtree(cache_dir)
        g = Graph(nodes, cache_dir=cache_dir)
        data = {"x": 1, "y": 2, "z": 2}

        all_nodes = [n.name for n in nodes]
        self.assertRaises(FileNotFoundError, g.calculate, data, from_cache=all_nodes, parallel=self.run_parallel)

    def test_node_exception_is_raised_by_graph_calculate(self):
        nodes = [
            Node(["x"], zero_division, "zero_division", ["output"])
        ]
        g = Graph(nodes)
        self.assertRaises(ZeroDivisionError, g.calculate, {"x": 1}, parallel=self.run_parallel)


class TestOperations(BaseTest):

    def test_add_and_mul(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add/output", "z"], mul, "mul", ["output"]),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data, parallel=self.run_parallel)

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
        results = g.calculate(data, parallel=self.run_parallel)

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
        results = g.calculate(data, parallel=self.run_parallel)

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
        results = g.calculate(data, parallel=self.run_parallel)

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
        results = g.calculate(parallel=self.run_parallel)

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
        results = g.calculate(data, parallel=self.run_parallel)

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

    def test_no_output(self):
        nodes = [
            Node(["x", "y"], add, "add"),
            Node(["y", "z"], mul, "mul"),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data, parallel=self.run_parallel)
        self.assertEqual(len(results), 0)

    def test_no_output2(self):
        nodes = [
            Node(["x", "y"], add_return_none, "add1"),
            Node(["y", "z"], add_return_none, "add2"),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data, parallel=self.run_parallel)
        self.assertEqual(len(results), 0)

    def test_multiple_non_connected_graphs(self):
        nodes = [
            Node(["add1/output", "x1"], add, "add2", ["output"]),
            Node(["x1", "x2"], add, "add1", ["output"]),
            Node(["x3", "x4"], mul, "mul", ["output"]),
        ]
        graph = Graph(nodes)

        data = {"x1": 5, "x2": 3, "x3": 3, "x4": 2}
        results = graph.calculate(data, parallel=self.run_parallel)

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
        results = g.calculate(data, parallel=self.run_parallel)

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
        results = g.calculate(data, parallel=self.run_parallel)

        self.assertEqual(len(results), 3)
        self.assertEqual(results["add1/output"], 1 + 2)
        self.assertEqual(results["mul/output"], (1 + 2) * 2)
        self.assertEqual(results["add2/output"], (1 + 2) + 2)

    def test_multiple_outputs(self):
        nodes = [
            Node(["x", "y"], add_subtract, "add_subtract", ["add_output", "subtract_output"]),
            Node(["add_subtract/add_output", "z"], mul, "mul", ["output"]),
        ]
        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data, parallel=self.run_parallel)

        self.assertEqual(len(results), 3)
        self.assertEqual(results["add_subtract/add_output"], 1 + 2)
        self.assertEqual(results["add_subtract/subtract_output"], 1 - 2)
        self.assertEqual(results["mul/output"], (1 + 2) * 2)

    def test_cache(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
            Node(["add/output", "z"], mul, "mul", ["output"]),
        ]

        cache_dir = "test_cache_dir"
        if exists(cache_dir):
            shutil.rmtree(cache_dir)
        g = Graph(nodes, cache_dir=cache_dir)
        data = {"x": 1, "y": 2, "z": 2}
        all_nodes = [n.name for n in nodes]

        # First let's verify that we cannot read data from cache
        self.assertRaises(FileNotFoundError, g.calculate, data, from_cache=all_nodes, parallel=self.run_parallel)

        # Then, write to cache output of all nodes
        results_ref = g.calculate(data, to_cache=all_nodes, parallel=self.run_parallel)

        # Check that we get the same results when reading different data from cache
        results1 = g.calculate(data, from_cache=all_nodes, parallel=self.run_parallel)
        results2 = g.calculate(data, from_cache=["add"], parallel=self.run_parallel)
        results3 = g.calculate(data, from_cache=["mul"], parallel=self.run_parallel)
        self.assertEqual(results1, results_ref)
        self.assertEqual(results2, results_ref)
        self.assertEqual(results3, results_ref)

        # Change input data
        data = {"x": 2, "y": 3, "z": 3}

        # When reading from cache, we should get the same results
        results4 = g.calculate(data, from_cache=all_nodes, parallel=self.run_parallel)
        self.assertEqual(results4, results_ref)

        # Without cache, we should get different results
        results5 = g.calculate(data, parallel=self.run_parallel)
        self.assertNotEquals(results5, results_ref)

    def test_lists(self):
        nodes = [
            Node(["x", "y"], add_lists, "add", ["output"]),
            Node(["add/output", "z"], mul_lists, "mul", ["output"]),
        ]
        g = Graph(nodes)

        size = 10
        x = size*[0]
        y = size*[1]
        z = size*[2]
        data = {"x": x, "y": y, "z": z}
        results = g.calculate(data, parallel=self.run_parallel)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add/output"], size*[1])
        self.assertEqual(results["mul/output"], size*[2])

    def test_large_lists(self):
        # Noticed that multiprocessing can get stuck with large inputs if not done correctly.
        # See e.g. https://stackoverflow.com/questions/59951832/python-multiprocessing-queue-makes-code-hang-with-large-data
        # This test will check that this doesn't happen
        nodes = [
            Node(["x", "y"], add_lists, "add", ["output"]),
            Node(["add/output", "z"], mul_lists, "mul", ["output"]),
        ]
        g = Graph(nodes)

        size = 10**6
        x = size*[0]
        y = size*[1]
        z = size*[2]
        data = {"x": x, "y": y, "z": z}
        results = g.calculate(data, parallel=self.run_parallel)

        self.assertEqual(len(results), len(nodes))
        self.assertEqual(results["add/output"], size*[1])
        self.assertEqual(results["mul/output"], size*[2])

    def test_large_graph(self):
        nodes = [
            Node(["x", "y"], add, "add1", ["output"]),
            Node(["add1/output", "z"], mul, "mul2", ["output"]),
            Node(["add1/output", "z"], div, "div3", ["output"]),
            Node(["mul2/output", "div3/output"], mul, "mul4", ["output"]),
            Node(["add1/output", "mul2/output"], add, "add5", ["output"]),
            Node(["add5/output", "div3/output"], mul, "mul6", ["output"]),
            Node(["add1/output", "z"], add, "add7", ["output"]),
            Node(["add7/output", "mul2/output"], div, "div8", ["output"]),
            Node(["add1/output", "mul4/output"], add, "add9", ["output"]),
            Node(["mul6/output", "div8/output"], mul, "mul10", ["output"]),
            Node(["add1/output", "mul2/output"], div, "div11", ["output"]),
            Node(["add7/output", "div3/output"], mul, "mul12", ["output"]),
            Node(["add1/output", "z"], add, "add13", ["output"]),
            Node(["add9/output", "mul6/output"], div, "div14", ["output"]),
            Node(["add1/output", "mul4/output"], add, "add15", ["output"]),
            Node(["mul2/output", "div8/output"], mul, "mul16", ["output"]),
            Node(["add13/output", "mul10/output"], add, "add17", ["output"]),
            Node(["mul16/output", "div14/output"], mul, "mul18", ["output"]),
            Node(["add15/output", "div14/output"], div, "div19", ["output"]),
            Node(["add9/output", "div8/output"], add, "add20", ["output"]),
            Node(["add1/output", "mul18/output"], add, "add21", ["output"]),
            Node(["add7/output", "y"], mul, "mul22", ["output"]),
            Node(["x", "z"], add, "add23", ["output"]),
            Node(["add21/output", "mul22/output"], div, "div24", ["output"]),
            Node(["add21/output", "z"], mul, "mul25", ["output"]),
            Node(["mul25/output", "div24/output"], mul, "mul26", ["output"]),
            Node(["y", "add1/output"], add, "add27", ["output"]),
            Node(["add27/output", "x"], mul, "mul28", ["output"]),
            Node(["y", "z"], add, "add29", ["output"]),
            Node(["add29/output", "x"], div, "div30", ["output"]),
            Node(["y", "mul26/output"], add, "add31", ["output"]),
            Node(["x", "y"], mul, "mul32", ["output"]),
            Node(["y", "x"], div, "div33", ["output"]),
            Node(["y", "z"], add, "add34", ["output"]),
            Node(["y", "y"], add, "add35", ["output"]),
            Node(["y", "mul32/output"], mul, "mul36", ["output"]),
            Node(["mul25/output", "div24/output"], div, "div37", ["output"]),
            Node(["add13/output", "mul10/output"], mul, "mul38", ["output"]),
            Node(["x", "div24/output"], add, "add39", ["output"]),
            Node(["y", "mul36/output"], div, "div40", ["output"]),
            Node(["mul38/output", "div40/output"], mul, "mul41", ["output"]),
            Node(["y", "mul36/output"], add, "add42", ["output"]),
            Node(["mul41/output", "div40/output"], div, "div43", ["output"]),
            Node(["y", "mul36/output"], add, "add44", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul45", ["output"]),
            Node(["y", "mul36/output"], add, "add46", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div47", ["output"]),
            Node(["y", "mul36/output"], add, "add48", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul49", ["output"]),
            Node(["y", "mul36/output"], add, "add50", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div51", ["output"]),
            Node(["y", "mul36/output"], add, "add52", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul53", ["output"]),
            Node(["y", "mul36/output"], add, "add54", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div55", ["output"]),
            Node(["y", "mul36/output"], add, "add56", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul57", ["output"]),
            Node(["y", "mul36/output"], add, "add58", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div59", ["output"]),
            Node(["y", "mul36/output"], add, "add60", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul61", ["output"]),
            Node(["y", "mul36/output"], add, "add62", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div63", ["output"]),
            Node(["y", "mul36/output"], add, "add64", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul65", ["output"]),
            Node(["y", "mul36/output"], add, "add66", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div67", ["output"]),
            Node(["y", "mul36/output"], add, "add68", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul69", ["output"]),
            Node(["y", "mul36/output"], add, "add70", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div71", ["output"]),
            Node(["y", "mul36/output"], add, "add72", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul73", ["output"]),
            Node(["y", "mul36/output"], add, "add74", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div75", ["output"]),
            Node(["y", "mul36/output"], add, "add76", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul77", ["output"]),
            Node(["y", "mul36/output"], add, "add78", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div79", ["output"]),
            Node(["y", "mul36/output"], add, "add80", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul81", ["output"]),
            Node(["y", "mul36/output"], add, "add82", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div83", ["output"]),
            Node(["y", "mul36/output"], add, "add84", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul85", ["output"]),
            Node(["y", "mul36/output"], add, "add86", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div87", ["output"]),
            Node(["y", "mul36/output"], add, "add88", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul89", ["output"]),
            Node(["y", "mul36/output"], add, "add90", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div91", ["output"]),
            Node(["y", "mul36/output"], add, "add92", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul93", ["output"]),
            Node(["y", "mul36/output"], add, "add94", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div95", ["output"]),
            Node(["y", "mul36/output"], add, "add96", ["output"]),
            Node(["mul41/output", "div43/output"], mul, "mul97", ["output"]),
            Node(["y", "mul36/output"], add, "add98", ["output"]),
            Node(["mul41/output", "div43/output"], div, "div99", ["output"]),
            Node(["y", "mul36/output"], add, "add100", ["output"]),
        ]

        g = Graph(nodes)
        data = {"x": 1, "y": 2, "z": 2}
        results = g.calculate(data, parallel=self.run_parallel)
        self.assertEqual(len(results), len(nodes))


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
