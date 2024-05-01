import logging
import time
from typing import List, Callable, Optional

from tinydag.exceptions import InvalidNodeFunctionOutput

logger = logging.getLogger(__name__)


class Node:
    """
    Nodes are used to define the graph. Node is defined by
    - list of needed inputs
    - functions that takes the defined inputs as input: func(*inputs)
    - node name
    - list of outputs

    function needs to return dict with the same keys as listed in outputs.

    E.g. Node(["x1", "x2"], add, "add2"), ["output"]] defines a node that
    - takes inputs "x1" and "x2"; this can be input data given by user or output of some other node
    - uses function add to calculate output of the node: results = add(x1, x2)
    - has name "add2"
    - has one output: "output
    function add should return a dict with a key: "output"
    """

    def __init__(self,
                 inputs: List[str],
                 function: Callable,
                 name: str,
                 outputs: Optional[List[str]] = None) -> None:
        """
        :param inputs: List of input names.
        :param function: Function that is used to calculate output of the node: output = function(*inputs)
        :param name: Name of the node.
        :param outputs: List of output names. If not given, then the function shouldn't return anything.
        """
        self.function = function
        self.inputs = inputs
        self.name = name
        if outputs is None:
            self.outputs = []
        else:
            self.outputs = [f"{self.name}/{item}" for item in outputs]

    def run(self, data: list) -> Optional[dict]:
        """
        Execute node.
        :param data: list of items that will be passed to function: function(*data).
        :return: Output of the node.
        """
        t_node_start = time.time()
        output = self.function(*data)
        t_node_end = time.time()
        logger.debug(f"Node {self.name} execution took {1000 * (t_node_end - t_node_start): 0.3f} ms")
        if len(self.outputs) == 0:
            return
        if not isinstance(output, dict):
            raise InvalidNodeFunctionOutput(f"Node {self.name} output is not a dict!")
        output = {f"{self.name}/{key}": val for key, val in output.items()}
        for item in self.outputs:
            if item not in output:
                raise InvalidNodeFunctionOutput(
                    f"Node {self.name} function output doesn't contain required item {item}!")
        return output

    def __repr__(self) -> str:
        return self.name
