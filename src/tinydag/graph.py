import logging
import os
from typing import List, Union, Optional

from tinydag.exceptions import InvalidGraphError, MissingInputError
from tinydag.internal.node_runner import NodeRunner
from tinydag.internal.rendering import render
from tinydag.node import Node

logger = logging.getLogger(__name__)


class Graph:
    """
    Directed and acyclic graph structure to orchestrate function calls.

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
                 cache_dir: str = "cache") -> None:
        """
        :param nodes: List of nodes defining the graph.
        :param cache_dir: Directory to save and read cached files.
        :raises InvalidGraphError if the node names are not unique.
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self._validate_node_names(nodes)
        self._nodes = nodes
        self._required_user_inputs = self._get_required_user_inputs()
        self._node_runner = NodeRunner(nodes, cache_dir)

    def render(self,
               path: str = "graph.gv",
               view: bool = True,
               show_outputs: bool = True) -> None:
        """
        Render graph. This will only work if graphviz is available.
        :param path: Path to save fig.
        :param view: Show graph fig.
        :param show_outputs: Show outputs of the graph.
        """
        try:
            render(self._nodes, self._required_user_inputs, path, view, show_outputs)
        except Exception as e:
            logger.warning(f"Graph cannot be rendered, caught error: {e}")
            return None

    def validate_graph(self) -> None:
        """
        Check if the graph structure is valid.
        :raises InvalidGraphError if the graph structure is not valid.
        """
        logger.info(f"Graph validation started")
        input_data = {name: None for name in self._required_user_inputs}
        self._node_runner.run(input_data, parallel=False, dry_run=True)
        logger.info(f"Graph validation was successful")

    def calculate(self,
                  input_data: Optional[dict] = None,
                  from_cache: Optional[List[str]] = None,
                  to_cache: Optional[List[str]] = None,
                  parallel: bool = False,
                  copy_node_input_data: bool = True) -> dict:
        """
        Execute every node in the graph.
        :param input_data: Input data for the graph, where keys are names used in the graph definition.
        :param from_cache: List of node names to read from cache.
        :param to_cache: List of node names to save to cache.
        :param parallel: Run nodes in parallel, one process for each.
        :param copy_node_input_data: Make deepcopy of the data that is passed to node.
        :return: Output of every node, with node outputs as keys.
        :raises MissingInputError if the input_data doesn't contain all the required data.
        :raises InvalidGraphError if the graph structure is not valid.
        :raises InvalidNodeFunctionOutput if the node function output is not valid.
        :raises FileNotFoundError if cache file we want to read doesn't exist.
        """
        self._validate_input_data(input_data)
        self.validate_graph()
        logger.info(f"Graph calculation started")
        return self._node_runner.run(input_data, from_cache, to_cache, parallel, copy_node_input_data, False)

    def __add__(self, nodes: Union[List[Node], Node]) -> "Graph":
        if isinstance(nodes, list):
            nodes = self._nodes + nodes
        else:
            nodes = self._nodes + [nodes]
        return Graph(nodes)

    def __repr__(self) -> str:
        repr_str = "\n"
        for node in self._nodes:
            name = node.name
            repr_str += f"Node: {name}\n"
            repr_str += "├─ Inputs:\n"
            for input_node in node.inputs:
                repr_str += f"│  ├─ {input_node}\n"
            repr_str += "└─ Outputs:\n"
            for output_node in node.outputs:
                repr_str += f"   ├─ {output_node}\n"
        return repr_str

    def _validate_input_data(self, input_data: Optional[dict]) -> None:
        if len(self._required_user_inputs) > 0:
            for item in self._required_user_inputs:
                if item not in input_data:
                    raise MissingInputError(f"Input data is missing {item}")

    def _get_required_user_inputs(self) -> List[str]:
        required_inputs, node_outputs = [], []
        for node in self._nodes:
            required_inputs += node.inputs
            node_outputs += node.outputs
        required_user_input = list(set(required_inputs) - set(node_outputs))
        logger.debug(f"Required user input: {required_user_input}")
        return required_user_input

    @staticmethod
    def _validate_node_names(nodes: List[Node]) -> None:
        node_names = [n.name for n in nodes]
        if len(set(node_names)) < len(node_names):
            raise InvalidGraphError("All the nodes need to have unique name!")
