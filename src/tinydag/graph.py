import logging
import os
import time
from copy import copy
from os.path import join
from typing import List, Callable, Union, Optional

from tinydag.exceptions import InvalidGraphError, MissingInputError, InvalidNodeFunctionOutput
from tinydag.node import Node
from tinydag.utils import load_pickle, save_pickle

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

    Defines a graph with following connections:

    x, y -> add1
    x, z -> add_subtract
    add1/output, x -> add2
    add1/output, add2/output -> mul
    mul/output, add_subtract/add_output -> div

    User needs to provide x, y and z as input data for this graph when doing calculation.

    Cache can be used to save and load cached results.
    """

    def __init__(self,
                 nodes: List[Node],
                 wrappers: Optional[List[Callable]] = None,
                 cache_dir="cache") -> None:
        """
        :param nodes: List of nodes defining the graph.
        :param wrappers: Optional wrapper functions that will be used to wrap all the node functions.
        :param cache_dir: Directory to save and read cached files.
        :raises InvalidGraphError if the node names are not unique.
        """
        self._check_node_names_are_unique(nodes)
        self.nodes = nodes
        self.wrappers = wrappers
        self.required_user_inputs = self._get_required_user_inputs()
        logger.debug(f"Required user input: {self.required_user_inputs}")

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

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
                dot.node(node.name, node.name, shape='box', style='filled', fillcolor='lightblue')
                for output in node.outputs:
                    dot.node(output, output, shape='oval', style='filled', fillcolor='lightgreen')
                    dot.edge(node.name, output)
                for node_input in node.inputs:
                    if node_input in self.required_user_inputs:
                        dot.node(node_input, node_input, shape='ellipse', style='filled', fillcolor='lightpink')
                    else:
                        dot.node(node_input, node_input, shape='oval', style='filled', fillcolor='lightgreen')
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

    def calculate(self,
                  input_data: Optional[dict] = None,
                  from_cache: Optional[List[str]] = None,
                  to_cache: Optional[List[str]] = None) -> dict:
        """
        Execute every node in the graph.
        :param input_data: Input data for the graph, where keys are names used in the graph definition.
        :param from_cache: List of node names to read from cache.
        :param to_cache: List of node names to save to cache.
        :return: Output of every node, with node outputs as keys.
        :raises MissingInputError if the input_data doesn't contain all the required data.
        :raises InvalidGraphError if the graph structure is not valid.
        :raises InvalidNodeFunctionOutput if the node function output is not valid.
        :raises FileNotFoundError if cache file we want to read doesn't exist.
        """
        self._check_input_data(input_data)
        return self._execute(input_data, True, from_cache, to_cache)

    def __add__(self, nodes: Union[List[Node], Node]) -> "Graph":
        if isinstance(nodes, list):
            nodes = self.nodes + nodes
        else:
            nodes = self.nodes + [nodes]
        return Graph(nodes, self.wrappers)

    def __repr__(self) -> str:
        repr_str = "\n"
        for node in self.nodes:
            name = node.name
            repr_str += f"Node: {name}\n"
            repr_str += "├─ Inputs:\n"
            for input_node in node.inputs:
                repr_str += f"│  ├─ {input_node}\n"
            repr_str += "└─ Outputs:\n"
            for output_node in node.outputs:
                repr_str += f"   ├─ {output_node}\n"
        return repr_str

    def _check_input_data(self, input_data):
        if len(self.required_user_inputs) > 0:
            for item in self.required_user_inputs:
                if item not in input_data:
                    raise MissingInputError(f"Input data is missing {item}")

    def _get_required_user_inputs(self) -> List[str]:
        required_inputs, node_outputs = [], []
        for node in self.nodes:
            required_inputs += node.inputs
            node_outputs += node.outputs
        return list(set(required_inputs) - set(node_outputs))

    def _execute(self,
                 input_data: Optional[dict] = None,
                 run: Optional[bool] = True,
                 from_cache=None,
                 to_cache=None) -> dict:

        if from_cache is None:
            from_cache = []
        if to_cache is None:
            to_cache = []

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
                if run:
                    results = self._get_node_results(node, node_input_data, from_cache, to_cache)
                    if results is not None:
                        for key, val in results.items():
                            inputs[node.name + "/" + key] = val
                else:
                    for output in node.outputs:
                        inputs[output] = None
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

        results = self._create_output(inputs)

        return results

    def _create_output(self, inputs: dict) -> dict:
        results = {}
        for node in self.nodes:
            for output in node.outputs:
                results[output] = inputs[output]
        return results

    def _get_node_results(self, node: Node, node_input_data: list, from_cache: List[str], to_cache: List[str]) -> dict:
        path = join(self.cache_dir, node.name)
        if node.name in from_cache:
            results = load_pickle(path)
            logger.info(f"Node {node.name} results read from cache: {path}")
        else:
            results = self._run_node(node, node_input_data)
        self._check_node_output(results, node)
        if node.name in to_cache:
            save_pickle(path, results)
            logger.info(f"Node {node.name} results wrote to cache: {path}")
        return results

    def _run_node(self, node: Node, data: list) -> dict:
        func = self._wrap_node_func(node.function)
        t_node_start = time.time()
        output = func(*data)
        t_node_end = time.time()
        logger.debug(f"Node {node} execution took {1000 * (t_node_end - t_node_start): 0.3f} ms")
        return output

    @staticmethod
    def _check_node_output(output: dict, node: Node) -> None:
        if output is not None:
            if not isinstance(output, dict):
                raise InvalidNodeFunctionOutput(f"Node {node.name} output is not a dict!")
        for item in node.outputs:
            item = item.replace(f"{node.name}/", "")
            if item not in output:
                raise InvalidNodeFunctionOutput(f"Node {node.name} output doesn't contain required item {item}!")

    def _wrap_node_func(self, func: Callable) -> Callable:
        if self.wrappers is not None:
            for wrapper in self.wrappers:
                func = wrapper(func)
        return func

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
