import concurrent.futures
import timeit
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread, Timer
from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union
from time import sleep, time

import numpy as np
import pandas as pd

import h5py

import csv
import pickle

from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from async_history import AsyncHistory
from async_strategy import AsynchronousStrategy

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    parameters_to_ndarrays
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.server import Server

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

class AsyncServer(Server):
    """Flower async server."""

    def __init__(
        self, *, client_manager: ClientManager, strategy: Strategy = None,  total_train_time: int = 85, waiting_interval: int = 5, max_workers: Optional[int] = 2, async_strategy: AsynchronousStrategy,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy
        self.async_strategy = async_strategy
        self.max_workers: Optional[int] = max_workers
        self.total_train_time = total_train_time
        self.waiting_interval = waiting_interval
        self.start_timestamp = 0.0
        self.end_timestamp = 0.0
        self.current_age = 0

    def set_new_params(self, new_params: Parameters):
        lock = Lock()
        with lock:
            self.parameters = new_params
            self.current_age += 1

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = AsyncHistory()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(timestamp=time(), loss=res[0])
            history.add_metrics_centralized(timestamp=time(), metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        start_time = time()
        end_timestamp = time() + self.total_train_time
        self.end_timestamp = end_timestamp
        self.start_timestamp = time()
        counter = 1
        self.fit_round(
            server_round=0,
            timeout=timeout,
            executor=executor,
            end_timestamp=end_timestamp,
            history=history
        )
        while time() - start_time < self.total_train_time:
            # If the clients are to be started periodically, move fit_round here and remove the executor.submit lines from _handle_finished_future_after_fit
            sleep(self.waiting_interval)
            self.evaluate_centralized(counter, history)
            counter += 1

        executor.shutdown(wait=True, cancel_futures=True)
        log(INFO, "FL finished")
        end_time = time()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
    
    def evaluate_centralized(self, current_round: int, history: History):
        res_cen = self.strategy.evaluate(
            current_round, parameters=self.parameters)
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            metrics_cen['end_timestamp'] = self.end_timestamp
            metrics_cen['start_timestamp'] = self.start_timestamp
            history.add_loss_centralized(
                timestamp=time(), loss=loss_cen)
            history.add_metrics_centralized(
                timestamp=time(), metrics=metrics_cen
            )

    def evaluate_centralized_async(self, history: AsyncHistory, parameters):
        res_cen = self.strategy.evaluate(
            0, parameters=parameters)
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            print("Accuracy: ", metrics_cen)
            metrics_cen['end_timestamp'] = self.end_timestamp
            metrics_cen['start_timestamp'] = self.start_timestamp
            history.add_loss_centralized_async(
                timestamp=time(), loss=loss_cen)
            history.add_metrics_centralized_async(
                timestamp=time(), metrics=metrics_cen
            )

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
        executor: ThreadPoolExecutor,
        end_timestamp: float,
        history: AsyncHistory,
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager
        )
        
        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        fit_clients(
            client_instructions=client_instructions,
            timeout=timeout,
            server=self,
            executor=executor,
            end_timestamp=end_timestamp,
            history=history,
        )

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    timeout: Optional[float],
    server: AsyncServer,
    executor: ThreadPoolExecutor,
    end_timestamp: float,
    history: AsyncHistory,
):
    """Refine parameters concurrently on all selected clients."""

    submitted_fs = {
        executor.submit(fit_client, client_proxy, ins, timeout)
        for client_proxy, ins in client_instructions
    }
    for f in submitted_fs:
        f.add_done_callback(
            lambda ftr: _handle_finished_future_after_fit(ftr, server=server, executor=executor, end_timestamp=end_timestamp, history=history),
        )


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,
    server: AsyncServer,
    executor: ThreadPoolExecutor,
    end_timestamp: float,
    history: AsyncHistory,
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    try:
        failure = future.exception()
    except concurrent.futures.CancelledError:
        print("Future was cancelled.")
        failure = ""
        return
    if failure is not None:
        print("Got a failure :(")
        return

    print("Got a result :)")
    result: Tuple[ClientProxy, FitRes] = future.result()
    clientProxy, res = result

    # Check result status code
    if res.status.code == Code.OK:
        parameters_aggregated = server.async_strategy.average(
            server.parameters, res.parameters, server.current_age - res.metrics['server_age'], res.num_examples
        )
        server.set_new_params(parameters_aggregated)
        history.add_metrics_distributed_fit_async(
            clientProxy.cid,{"sample_sizes": res.num_examples, **res.metrics }, timestamp=time()
        )
        data_to_save = (server.current_age, parameters_to_ndarrays(parameters_aggregated))
        if(server.current_age < 151):
            with open('parameters.bin', 'ab') as file:
                pickle.dump(data_to_save, file)
        else:
            print("DONE")


    
    #if time() < end_timestamp:
    if not executor._shutdown:
        log(DEBUG, f"Yippie! Starting the client {clientProxy.cid} again \U0001f973")
        new_ins = FitIns(server.parameters, {
            "lr": 0.03,
            "momentum": 0.9,
            "local_epochs": 4,
            "server_age": server.current_age 
        })
        ftr = executor.submit(fit_client, client=clientProxy, ins=new_ins, timeout=None)
        ftr.add_done_callback(lambda ftr: _handle_finished_future_after_fit(ftr, server, executor, end_timestamp, history))


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
