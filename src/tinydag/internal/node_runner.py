import logging
import multiprocessing
import time
from copy import deepcopy
from multiprocessing import Queue, Process
from os.path import join
from typing import List, Optional, Dict, Tuple

from tinydag.exceptions import InvalidGraphError
from tinydag.node import Node
from tinydag.internal.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)


class NodeRunner:
    """
    Internal class to run nodes.
    """

    def __init__(self,
                 nodes: List[Node],
                 cache_dir: str = "cache") -> None:
        self._cache_dir = cache_dir
        self._nodes = nodes
        self._from_cache = None
        self._to_cache = None
        self._parallel = False
        self._copy_node_input_data = True
        self._dry_run = True

    def run(self,
            input_data: Optional[dict] = None,
            from_cache: Optional[List[str]] = None,
            to_cache: Optional[List[str]] = None,
            parallel: bool = False,
            copy_node_input_data: bool = True,
            dry_run: bool = True) -> dict:
        self._from_cache = from_cache if from_cache is not None else []
        self._to_cache = to_cache if to_cache is not None else []
        self._parallel = parallel
        self._copy_node_input_data = copy_node_input_data
        self._dry_run = dry_run
        logger.info(f"Graph calculation started")
        return self._execute(input_data)

    def _execute(self, input_data: Optional[dict] = None) -> dict:
        t_graph_start = time.time()
        # TODO: refactor methods for parallel and sequential processing, now they contain plenty of duplicate logic and code
        if self._parallel:
            outputs = self._run_nodes_parallel(input_data)
        else:
            outputs = self._run_nodes_sequentially(input_data)
        logger.info("All nodes executed successfully")
        t_graph_end = time.time()
        logger.debug(f"Graph execution took {1000 * (t_graph_end - t_graph_start): 0.2f} ms")
        return self._create_output(outputs)

    def _run_nodes_sequentially(self, input_data: Optional[dict]) -> dict:
        nodes_to_execute = [i for i in range(len(self._nodes))]

        # Container where all the node inputs will be stored
        # This will be updated when the nodes are executed
        inputs = deepcopy(input_data) if input_data is not None else {}

        # Loop until all the nodes are executed
        while len(nodes_to_execute) > 0:

            # Execute every node that has all the inputs available
            nodes_executed, inputs = self._execute_nodes_with_inputs_available(inputs, nodes_to_execute)

            # Check that at least one of the nodes has been executed during this round
            # If not, it means that the graph has invalid structure
            if len(nodes_executed) == 0:
                raise InvalidGraphError("Graph cannot be executed! The graph has invalid structure.")

            for node_index in nodes_executed:
                nodes_to_execute.remove(node_index)

            logger.info(
                f"Number of nodes: \n " +
                f"unfinished: {len(nodes_to_execute)} \n " +
                f"finished: {len(self._nodes) - len(nodes_to_execute)}")

        return inputs

    def _execute_nodes_with_inputs_available(self,
                                             inputs: dict,
                                             nodes_to_execute: List[int]) -> Tuple[List[int], dict]:
        nodes_executed = []
        for node_index in nodes_to_execute:
            node = self._nodes[node_index]
            node_input_data = self._collect_node_input_data(node, inputs)
            if len(node_input_data) < len(node.inputs):
                continue
            if self._dry_run:
                for output in node.outputs:
                    inputs[output] = None
            else:
                results = self._run_node_and_cache(node, node_input_data)
                if results is not None:
                    inputs.update(results)
            nodes_executed.append(node_index)
        return nodes_executed, inputs

    def _run_nodes_parallel(self, input_data: Optional[dict]) -> dict:
        # Init variables shared between the processes
        manager = multiprocessing.Manager()
        inputs = manager.dict(input_data) \
            if input_data is not None \
            else manager.dict()
        exception_queue = multiprocessing.Queue()

        # Loop until all the nodes are executed
        nodes_to_execute = [i for i in range(len(self._nodes))]
        running_processes = {}
        while len(nodes_to_execute) > 0:
            # Start process for every node that has all the inputs available and is not already running
            running_processes = self._start_node_processes(inputs, nodes_to_execute, running_processes, exception_queue)

            # Wait until at least one of the running processes has finished
            node_index_finished = self._wait_process_to_finish(running_processes, exception_queue)
            nodes_to_execute.remove(node_index_finished)
            running_processes.pop(node_index_finished)
            logger.info(
                f"Number of nodes: \n " +
                f"in processing: {len(running_processes)} \n " +
                f"unfinished: {len(nodes_to_execute)} \n " +
                f"finished: {len(self._nodes) - len(nodes_to_execute)}")

        return inputs

    def _node_task(self,
                   node_index: int,
                   node_input_data: list,
                   inputs: dict,
                   exception_queue: Queue) -> None:
        node = self._nodes[node_index]
        logger.debug(f"Launched task for node {node}, process id {multiprocessing.current_process().pid}")
        try:
            results = self._run_node_and_cache(node, node_input_data)
            if results is not None:
                inputs.update(results)  # Lock is built-in, no need to do it manually
            return
        except Exception as e:
            logger.warning(f"Node {node} raised exception {e}")
            exception_queue.put(e)  # Lock is built-in, no need to do it manually
            return

    @staticmethod
    def _wait_process_to_finish(running_processes: Dict[int, Process], exception_queue: Queue) -> int:
        while True:
            for node_index, process in running_processes.items():
                if not process.is_alive():
                    logger.debug(f"Process {process.pid} has finished")
                    while not exception_queue.empty():
                        exception = exception_queue.get()
                        logger.warning(f"Received exception {exception}")
                        raise exception

                    return node_index

    def _start_node_processes(self,
                              inputs: dict,
                              nodes_to_execute: List[int],
                              running_processes: Dict[int, Process],
                              exception_queue: Queue) -> Dict[int, Process]:
        node_indices_to_check = set(nodes_to_execute) - set(running_processes.keys())
        for node_index in node_indices_to_check:
            node = self._nodes[node_index]
            node_input_data = self._collect_node_input_data(node, inputs)
            if len(node_input_data) < len(node.inputs):
                continue
            args = (node_index, node_input_data, inputs, exception_queue)
            process = multiprocessing.Process(target=self._node_task, args=args)
            process.start()
            running_processes[node_index] = process
        return running_processes

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
            logger.info(f"Node {node} executed successfully")
        if node.name in self._to_cache:
            save_pickle(path, results)
            logger.info(f"Node {node.name} results wrote to cache: {path}")
        return results

    @staticmethod
    def _collect_node_input_data(node: Node, inputs: dict) -> list:
        logger.debug(f"Collecting input for node {node}")
        input_data = []
        for i in node.inputs:
            if i in inputs:
                input_data.append(inputs[i])
            else:
                logger.debug(f"Cannot find input {i} for node {node}.")
                break  # We cannot execute node without full input, so no need to continue
        logger.debug(f"Found all the inputs for the node {node}.")
        return input_data
