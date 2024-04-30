import logging
import time
from copy import copy
from typing import List, Callable, Union, Optional, Any

from tinydag.exceptions import InvalidGraphError, MissingInputError
from tinydag.node import Node

logger = logging.getLogger(__name__)

try:
    import graphviz as graphviz
    from graphviz import Digraph
except ImportError:
    logger.warning("Cannot import graphviz")


class Graph:
    """
    Minimal implementation of computational (directed, acyclic) graph.

    User provides the graph structure (nodes) and input data for the graph. Every node waits until input data for that
    node is ready. Eventually, the graph executes every node in the graph and returns output of every node as the
    result.

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

    where add, mul and div are functions. This determines the graph where

    x,y -> add1
    add1,x -> add2
    add1,add2 -> mul
    mul,z -> div

    User needs to provide x, y and z as input data for this graph when doing calculation.
    """

    def __init__(self,
                 nodes: List[Node],
                 wrappers: Optional[List[Callable]] = None) -> None:
        """
        :param nodes: List of nodes defining the graph.
        :param wrappers: Optional wrapper functions that will be used to wrap all the node functions.
        :raises InvalidGraphError if the node names are not unique.
        """
        self._check_node_names_are_unique(nodes)
        self.nodes = nodes
        self.wrappers = wrappers
        self.required_user_inputs = self._get_required_user_inputs()
        logger.debug(f"Required user input: {self.required_user_inputs}")

    def _get_required_user_inputs(self) -> List[str]:
        node_names = [node.name for node in self.nodes]
        required_inputs = []
        for node in self.nodes:
            required_inputs += node.inputs
        return list(set(required_inputs) - set(node_names))

    def render(self,
               path: str = "graph.gv",
               view: bool = True) -> Optional["Digraph"]:
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

    def check(self) -> None:
        """
        Check if the graph structure is valid.
        :raises InvalidGraphError if the graph structure is not valid.
        """
        input_data = {name: None for name in self.required_user_inputs}
        self._execute(input_data, False)

    def calculate(self, input_data: Optional[dict] = None) -> dict:
        """
        Execute every node in the graph.
        :param input_data: Input data for the graph, where keys are names used in the graph definition.
        :return: Output of every node, with node names as keys.
        :raises MissingInputError if the input_data doesn't contain all the required data.
        :raises InvalidGraphError if the graph structure is not valid.
        """
        self._check_input_data(input_data)
        return self._execute(input_data)

    def _check_input_data(self, input_data):
        if len(self.required_user_inputs) > 0:
            for item in self.required_user_inputs:
                if item not in input_data:
                    raise MissingInputError(f"Input data is missing {item}")

    def _execute(self, input_data: Optional[dict] = None, run: Optional[bool] = True) -> dict:
        # Container where all the node inputs will be stored
        # This will be updated when the nodes are executed
        inputs = copy(input_data) if input_data is not None else {}

        nodes_to_execute = [i for i in range(len(self.nodes))]
        t_graph_start = time.time()

        # Loop until all the nodes are executed
        while len(nodes_to_execute) > 0:
            logger.debug(f"Nodes to execute: {nodes_to_execute}")

            # Execute every node that has all the inputs available
            nodes_executed = []
            for node_index in nodes_to_execute:
                node = self.nodes[node_index]
                logger.debug(f"Executing node {node}")
                node_input_data = self._get_node_input_data(node, inputs)
                if len(node_input_data) < len(node.inputs):
                    continue  # All the input data cannot be found for this node yet, so skip this node
                output = self._run_node(node, node_input_data) if run else "output"
                inputs[node.name] = output
                nodes_executed.append(node_index)
                logger.debug(f"Node {node} executed successfully")

            # Check that at least one of the nodes has been executed during this round
            # If not, it means that the graph has invalid structure
            if len(nodes_executed) == 0:
                raise InvalidGraphError("Graph cannot be executed! The graph has invalid structure.")

            for node_index in nodes_executed:
                nodes_to_execute.remove(node_index)

        logger.debug("All nodes executed successfully")
        t_graph_end = time.time()
        logger.debug(f"Graph execution took {1000 * (t_graph_end - t_graph_start): 0.2f} ms")

        # As a result we return node outputs
        # We get this from inputs, by removing the original input data from the dict
        results = inputs
        if input_data is not None:
            for key in input_data.keys():
                results.pop(key)

        return results

    def _run_node(self, node: Node, data: list) -> Any:
        func = self._wrap_node_func(node.function)
        t_node_start = time.time()
        output = func(*data)
        t_node_end = time.time()
        logger.debug(f"Node {node} execution took {1000 * (t_node_end - t_node_start): 0.3f} ms")
        return output

    def _wrap_node_func(self, func):
        if self.wrappers is not None:
            for wrapper in self.wrappers:
                func = wrapper(func)
        return func

    def __add__(self, nodes: Union[List[Node], Node]) -> "Graph":
        if isinstance(nodes, list):
            nodes = self.nodes + nodes
        else:
            nodes = self.nodes + [nodes]
        return Graph(nodes, self.wrappers)

    def __repr__(self) -> str:
        return str([n.name for n in self.nodes])

    @staticmethod
    def _check_node_names_are_unique(nodes: List[Node]) -> None:
        node_names = [n.name for n in nodes]
        if len(set(node_names)) < len(node_names):
            raise InvalidGraphError("All the nodes need to have unique name!")

    @staticmethod
    def _get_node_input_data(node: Node, inputs: dict) -> list:
        input_data = []
        for i in node.inputs:
            if i in inputs:
                input_data.append(inputs[i])
            else:
                logger.debug(f"Cannot find input {i} for node {node}.")
                break  # We cannot execute node without full input, so no need to continue
        return input_data
