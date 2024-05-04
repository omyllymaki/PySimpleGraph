import logging
import multiprocessing
import os
import time
from copy import copy, deepcopy
from os.path import join
from typing import List, Union, Optional

from tinydag.exceptions import InvalidGraphError, MissingInputError
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
                 cache_dir="cache") -> None:
        """
        :param nodes: List of nodes defining the graph.
        :param cache_dir: Directory to save and read cached files.
        :raises InvalidGraphError if the node names are not unique.
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._cache_dir = cache_dir

        self._validate_node_names(nodes)
        self._nodes = nodes
        self._required_user_inputs = self._get_required_user_inputs()
        self._from_cache = None
        self._to_cache = None
        self._parallel = False
        self._copy_node_input_data = True
        self._run_nodes = True

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
            for node in self._nodes:
                dot.node(node.name, node.name, shape='box', style='filled', fillcolor='lightblue')
                for output in node.outputs:
                    dot.node(output, output, shape='oval', style='filled', fillcolor='lightgreen')
                    dot.edge(node.name, output)
                for node_input in node.inputs:
                    if node_input in self._required_user_inputs:
                        dot.node(node_input, node_input, shape='ellipse', style='filled', fillcolor='lightpink')
                    else:
                        dot.node(node_input, node_input, shape='oval', style='filled', fillcolor='lightgreen')
                    dot.edge(node_input, node.name)
            dot.render(path, view=view)
            return dot
        except Exception as e:
            logger.warning(f"Graph cannot be rendered, caught error: {e}")
            return None

    def validate_graph(self) -> None:
        """
        Check if the graph structure is valid.
        :raises InvalidGraphError if the graph structure is not valid.
        """
        self._run_nodes = False
        input_data = {name: None for name in self._required_user_inputs}
        self._run_nodes_sequentially(input_data)

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

        self._from_cache = from_cache if from_cache is not None else []
        self._to_cache = to_cache if to_cache is not None else []
        self._parallel = parallel
        self._copy_node_input_data = copy_node_input_data
        self._run_nodes = True

        return self._execute(input_data)

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

    def _execute(self, input_data: Optional[dict] = None) -> dict:
        t_graph_start = time.time()
        # TODO: refactor methods for parallel and sequential processing, now they contain plenty of duplicate logic and code
        if self._parallel:
            outputs = self._run_nodes_parallel(input_data)
        else:
            outputs = self._run_nodes_sequentially(input_data)
        logger.debug("All nodes executed successfully")
        t_graph_end = time.time()
        logger.debug(f"Graph execution took {1000 * (t_graph_end - t_graph_start): 0.2f} ms")
        return self._create_output(outputs)

    def _run_nodes_sequentially(self, input_data):
        nodes_to_execute = [i for i in range(len(self._nodes))]

        # Container where all the node inputs will be stored
        # This will be updated when the nodes are executed
        inputs = deepcopy(input_data) if input_data is not None else {}

        # Loop until all the nodes are executed
        while len(nodes_to_execute) > 0:
            logger.debug(f"Nodes to execute: {nodes_to_execute}")

            # Execute every node that has all the inputs available
            nodes_executed = []
            for node_index in nodes_to_execute:
                node = self._nodes[node_index]
                logger.debug(f"Executing node {node}")
                node_input_data = self._collect_node_input_data(node, inputs)
                if len(node_input_data) < len(node.inputs):
                    logger.debug(f"Cannot find all the inputs for the node {node}.")
                    continue  # All the input data cannot be found for this node yet, so skip this node
                logger.debug(f"Found all the inputs for the node {node}.")
                if self._run_nodes:
                    results = self._run_node_and_cache(node, node_input_data)
                    if results is not None:
                        inputs.update(results)
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

        return inputs

    def _run_nodes_parallel(self, input_data):
        nodes_to_execute = [i for i in range(len(self._nodes))]
        manager = multiprocessing.Manager()
        inputs = manager.dict(input_data) if input_data is not None else manager.dict()
        queue = multiprocessing.Queue()
        exception_queue = multiprocessing.Queue()
        lock = multiprocessing.Lock()

        def node_task(node_index):
            node = self._nodes[node_index]
            logger.debug(f"Launched task for node {node}, process id {multiprocessing.current_process().pid}")
            try:
                # Wait until input data is available
                counter = 0
                while True:
                    logger.debug(f"Trying to execute node {node}")
                    node_input_data = self._collect_node_input_data(node, inputs)
                    if len(node_input_data) < len(node.inputs):
                        if counter % 100 == 0:
                            logger.debug(f"Cannot find all the inputs for the node {node}.")
                        time.sleep(0.0001)
                        counter += 1
                        continue
                    else:
                        break

                logger.debug(f"Found all the inputs for the node {node}.")
                results = self._run_node_and_cache(node, node_input_data)
                if results is not None:
                    with lock:
                        inputs.update(results)

                logger.debug(f"Node {node} executed successfully")
                logger.debug(f"Add node {node} output to inputs, inputs now contains {inputs.keys()}")
                queue.put(node_index)
            except Exception as e:
                logger.debug(f"Node {node} raised exception {e}")
                exception_queue.put(e)
                queue.put(None)

        processes = []
        for node_index in nodes_to_execute:
            process = multiprocessing.Process(target=node_task, args=(node_index,))
            process.start()
            processes.append(process)

        # Wait for all nodes to finish processing
        while True:
            completed_node_index = queue.get()
            if not exception_queue.empty():
                exception = exception_queue.get()
                logger.debug(f"Received exception {exception}")
                raise exception

            logger.debug(f"Completed node index {completed_node_index}")
            if completed_node_index is not None:
                nodes_to_execute.remove(completed_node_index)
                if len(nodes_to_execute) == 0:
                    break
        return inputs

    def _create_output(self, inputs: dict) -> dict:
        results = {}
        for node in self._nodes:
            for output in node.outputs:
                results[output] = inputs[output]
        return results

    def _run_node_and_cache(self,
                            node: Node,
                            node_input_data: list) -> Optional[dict]:
        path = join(self._cache_dir, node.name)
        if node.name in self._from_cache:
            results = load_pickle(path)
            logger.info(f"Node {node.name} results read from cache: {path}")
        else:
            if self._copy_node_input_data:
                results = node.run(deepcopy(node_input_data))
            else:
                results = node.run(node_input_data)
        if node.name in self._to_cache:
            save_pickle(path, results)
            logger.info(f"Node {node.name} results wrote to cache: {path}")
        return results

    @staticmethod
    def _validate_node_names(nodes: List[Node]) -> None:
        node_names = [n.name for n in nodes]
        if len(set(node_names)) < len(node_names):
            raise InvalidGraphError("All the nodes need to have unique name!")

    @staticmethod
    def _collect_node_input_data(node: Node, inputs: dict) -> list:
        input_data = []
        for i in node.inputs:
            if i in inputs:
                input_data.append(inputs[i])
            else:
                logger.debug(f"Cannot find input {i} for node {node}.")
                break  # We cannot execute node without full input, so no need to continue
        return input_data
