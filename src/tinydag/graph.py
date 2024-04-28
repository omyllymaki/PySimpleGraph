import logging
import time
from copy import copy
from typing import List, Callable, Union

from tinydag.node import Node

logger = logging.getLogger(__name__)

try:
    import graphviz as graphviz
    from graphviz import Digraph
except ImportError:
    logger.warning("Cannot import graphviz")


class GraphError(Exception):
    pass


class Graph:
    """
    Minimal implementation of computational (directed, acyclic) graph.

    User provides graph structure (nodes) and input data for graph. Every node waits until input data for that node
    is ready. Eventually, graph executes every node in graph and returns output of every node as result.
    """

    def __init__(self,
                 nodes: List[Node],
                 wrappers: List[Callable] = None):
        """
        :param nodes: List of nodes.
        :param wrappers: Optional wrapper functions that will be used to wrap all functions in nodes.

        User needs to specify inputs, function and name for every node.

        Example:

        add = lambda a, b: a + b
        mul = lambda a, b: a * b
        div = lambda a, b: a / b

        nodes = [
            Node(["add1", "x"], add, "add2"),
            Node(["add1", "add2"], mul, "mul"),
            Node(["x", "y"], add, "add1"),
            Node(["mul", "z"], div, "div"),
        ]

        where add, mul and div are functions. This determines graph where

        x,y -> add1
        add1,x -> add2
        add1,add2 -> mul
        mul,z -> div

        Note that user needs to provide x, y and z as input data for this graph when doing calculation.
        """

        self._check_nodes(nodes)
        self.nodes = nodes
        self.wrappers = wrappers

    def render(self,
               path: str = "graph.gv",
               view: bool = True) -> "Digraph":
        """
        Render graph. This will only work if graphviz is available.
        :param path: Path to save fig.
        :param view: Show graph fig.
        :return: graphviz Digraph is graphviz is available, otherwise None.
        """

        try:
            dot = graphviz.Digraph()
            for node in self.nodes:
                dot.node(node.name, node.name)
            for node in self.nodes:
                for node_input in node.inputs:
                    dot.edge(node_input, node.name)
            dot.render(path, view=view)
            return dot
        except Exception as e:
            logger.warning(f"Graph cannot be rendered, caught error: {e}")
            return None

    def check(self, input_data: dict = None) -> None:
        """
        Check if graph can be executed. Raises Exception if graph is not valid.

        :param input_data: Input data for graph, where keys are names used in graph definition.
        """
        self._execute(input_data, False)

    def calculate(self, input_data: dict = None) -> dict:
        """
        Execute every node in graph.
        :param input_data: Input data for graph, where keys are names used in graph definition.
        :return: Output of every node, with node names as keys.
        """
        return self._execute(input_data)

    def _execute(self, input_data: dict = None, run: bool = True) -> dict:
        # Container where all the node outputs will be stored
        results = copy(input_data) if input_data is not None else {}

        nodes_to_execute = [i for i in range(len(self.nodes))]
        t_graph_start = time.time()

        # Loop until all nodes are executed
        while len(nodes_to_execute) > 0:
            logger.debug(f"Nodes to execute: {nodes_to_execute}")

            # Execute every node that has all the inputs available
            nodes_executed = []
            for node_index in nodes_to_execute:
                node = self.nodes[node_index]
                logger.debug(f"Executing node {node}")
                node_input_data = self._get_input_data(node, results)
                if len(node_input_data) < len(node.inputs):
                    continue  # All the input data cannot be found for this node yet
                output = self._run_node(node, node_input_data) if run else "output"
                results[node.output] = output
                nodes_executed.append(node_index)
                logger.debug(f"Node {node} executed successfully")

            # Check that at least one of the nodes has been executed during this round
            # If not, then it means that graph has invalid struct or that all the input is not provided
            if len(nodes_executed) == 0:
                raise GraphError("Graph cannot be executed! One or multiple inputs are missing.")

            for node_index in nodes_executed:
                nodes_to_execute.remove(node_index)

        logger.debug("All nodes executed successfully")
        t_graph_end = time.time()
        logger.debug(f"Graph execution took {1000 * (t_graph_end - t_graph_start): 0.2f} ms")

        # Remove inputs
        if input_data is not None:
            for key in input_data.keys():
                results.pop(key)

        return results

    def _run_node(self, node, data):
        f = node.function
        if self.wrappers is not None:
            for wrapper in self.wrappers:
                f = wrapper(f)
        t_node_start = time.time()
        output = f(*data)
        t_node_end = time.time()
        logger.debug(f"Node {node} execution took {1000 * (t_node_end - t_node_start): 0.3f} ms")
        return output

    def __add__(self, nodes: Union[List[Node], Node]):
        if isinstance(nodes, list):
            nodes = self.nodes + nodes
        else:
            nodes = self.nodes + [nodes]
        return Graph(nodes, self.wrappers)

    def __repr__(self):
        return str([n.name for n in self.nodes])

    @staticmethod
    def _check_nodes(nodes):
        node_names = [n.name for n in nodes]
        if len(set(node_names)) < len(node_names):
            raise GraphError("All the nodes need to have unique name!")

    @staticmethod
    def _get_input_data(node, results):
        input_data = []
        for i in node.inputs:
            val = results.get(i, None)
            if val is None:
                logger.debug(f"Cannot find input {i} for node {node}.")
                break
            else:
                input_data.append(val)
        return input_data
