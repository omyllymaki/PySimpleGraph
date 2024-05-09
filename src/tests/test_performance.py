import functools
import inspect
import logging
import os
import shutil
import statistics
import time
import timeit
import unittest
from functools import partial
from os.path import exists

from tinydag.exceptions import InvalidGraphError, MissingInputError, InvalidNodeFunctionOutput
from tinydag.graph import Graph
from tinydag.node import Node

logging.basicConfig(level=logging.WARNING)


def add(a, b):
    return {"output": a + b}


def mul(a, b):
    return {"output": a * b}


def sleep(x):
    time.sleep(0.1)
    return {"output": None}


def add_lists(a, b):
    return {"output": [i + j for i, j in zip(a, b)]}


def mul_lists(a, b):
    return {"output": [i * j for i, j in zip(a, b)]}


def print_test_name_decorator(test_method):
    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        print("------")
        print("Current test running:", test_method.__name__)
        return test_method(self, *args, **kwargs)

    return wrapper


class BaseTest(unittest.TestCase):

    @staticmethod
    def print_statistics(execution_times):
        min_time = min(execution_times)
        max_time = max(execution_times)
        mean_time = statistics.mean(execution_times)
        stdev_time = statistics.stdev(execution_times)

        # Convert seconds to milliseconds
        min_time_ms = min_time * 1000
        max_time_ms = max_time * 1000
        mean_time_ms = mean_time * 1000
        stdev_time_ms = stdev_time * 1000

        print(f"Minimum execution time: {min_time_ms:.1f} ms")
        print(f"Maximum execution time: {max_time_ms:.1f} ms")
        print(f"Mean execution time: {mean_time_ms:.1f} ms")
        print(f"Standard deviation of execution times: {stdev_time_ms:.1f} ms")

    def check_performance(self, g, data, number=1, repeat=100):
        print("\nStart sequential calculation")
        func = partial(g.calculate, input_data=data, parallel=False)
        execution_times = timeit.repeat(func, number=number, repeat=repeat)
        self.print_statistics(execution_times)

        print("\nStart parallel calculation")
        func = partial(g.calculate, input_data=data, parallel=True)
        execution_times = timeit.repeat(func, number=number, repeat=repeat)
        self.print_statistics(execution_times)


class TestPerformance(BaseTest):

    @print_test_name_decorator
    def test_one_node(self):
        nodes = [
            Node(["x", "y"], add, "add", ["output"]),
        ]
        g = Graph(nodes)

        data = {"x": 1, "y": 2}
        self.check_performance(g, data, repeat=100)

    @print_test_name_decorator
    def test_parallel_sleep(self):
        nodes = [
            Node(["x"], sleep, "sleep1"),
            Node(["x"], sleep, "sleep2"),
            Node(["x"], sleep, "sleep3"),
            Node(["x"], sleep, "sleep4"),
            Node(["x"], sleep, "sleep5"),
        ]
        g = Graph(nodes)

        data = {"x": 1, "y": 2}
        self.check_performance(g, data, repeat=10)


if __name__ == '__main__':
    unittest.main()
