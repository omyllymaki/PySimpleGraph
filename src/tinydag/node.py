import functools
from typing import List, Callable, Optional


class Node:
    """
    Nodes are used to define the graph. Node is defined by
    - list of needed inputs
    - functions that takes the defined inputs as input: func(*inputs)
    - node name

    E.g. Node(["x1", "x2"], add, "add2")] defines a node that
    - takes inputs "x1" and "x2"; this can be input data given by user or output of some other node
    - uses function add to calculate output of the node: output = add(x1, x2)
    - has name "add2"
    """

    def __init__(self,
                 inputs: List[str],
                 function: Callable,
                 name: str) -> None:
        """
        :param inputs: List of input names.
        :param function: Function that is used to calculate output of the node: output = function(*inputs)
        :param name: Name of the node.
        """
        self.function = function
        self.inputs = inputs
        self.name = name

    def __repr__(self) -> str:
        return self.name
