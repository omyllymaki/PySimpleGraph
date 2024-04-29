import functools
from typing import List, Callable, Optional


class Node:
    """
    Nodes are used to define the graph. Node is defined by
    - list of needed inputs
    - functions that takes the defined inputs as input: func(*inputs)
    - node name
    - output name

    E.g. Node(["x1", "x2"], add, "add2")] defines a node that
    - takes inputs "x1" and "x2"; this can be input data given by user or output of some other node
    - uses function add to calculate output of the node: output = add(x1, x2)
    - has name "add2"
    """

    def __init__(self,
                 inputs: List[str],
                 function: Callable,
                 name: Optional[str] = None,
                 output_name: Optional[str] = None) -> None:
        """
        :param inputs: List of input names.
        :param function: Function that is used to calculate output of the node: output = function(*inputs)
        :param name: Optional name of the node.
        :param output_name: Optional name of the output.
        """
        self.function = function
        self.inputs = inputs
        self.name = name if name is not None else self.get_func_name(function)
        self.output_name = output_name if output_name is not None else self.name

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def get_func_name(f):
        if isinstance(f, (functools.partial, functools.partialmethod)):
            fname = f.func.__name__
        else:
            fname = f.__name__
        return fname
