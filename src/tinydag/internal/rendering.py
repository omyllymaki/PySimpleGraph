import logging
from typing import List

from tinydag.node import Node

logger = logging.getLogger(__name__)

try:
    import graphviz as graphviz
except ImportError:
    logger.warning("Cannot import graphviz")


def render(nodes: List[Node],
           inputs: List[str],
           path: str = "graph.gv",
           view: bool = True,
           show_outputs: bool = True):
    if show_outputs:
        _render_with_outputs(nodes, inputs, path, view)
    else:
        _render_without_outputs(nodes, inputs, path, view)


def _render_without_outputs(nodes: List[Node],
                            inputs: List[str],
                            path: str = "graph.gv",
                            view: bool = True) -> None:
    dot = graphviz.Digraph()
    for node in nodes:
        dot.node(node.name, node.name, shape='box', style='filled', fillcolor='lightblue')
        for node_input in node.inputs:
            if node_input in inputs:
                dot.node(node_input, node_input, shape='ellipse', style='filled', fillcolor='lightpink')
                dot.edge(node_input, node.name)

        for output_node in nodes:
            if output_node.name != node.name:
                for output in output_node.outputs:
                    if output in node.inputs:
                        dot.edge(output_node.name, node.name)

    dot.render(path, view=view)


def _render_with_outputs(nodes: List["Node"],
                         inputs: List[str],
                         path: str = "graph.gv",
                         view: bool = True) -> None:
    dot = graphviz.Digraph()
    for node in nodes:
        dot.node(node.name, node.name, shape='box', style='filled', fillcolor='lightblue')
        for output in node.outputs:
            dot.node(output, output, shape='oval', style='filled', fillcolor='lightgreen')
            dot.edge(node.name, output)
        for node_input in node.inputs:
            if node_input in inputs:
                dot.node(node_input, node_input, shape='ellipse', style='filled', fillcolor='lightpink')
            else:
                dot.node(node_input, node_input, shape='oval', style='filled', fillcolor='lightgreen')
            dot.edge(node_input, node.name)
    dot.render(path, view=view)
