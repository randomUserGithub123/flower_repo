import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset, prepare_dataset_nonIID
from client import generate_client_fn
import flwr as fl
from server import get_on_fit_config, get_evaluate_fn

import matplotlib.pyplot as plt
import csv
import pickle
import h5py
from dataclasses import dataclass

from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from async_server import AsyncServer
from async_strategy import AsynchronousStrategy

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset_nonIID(cfg.num_clients, cfg.batch_size)
    print(len(trainloaders), len(trainloaders[0].dataset))

    ## 4. Define your strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  #

    fl.server.start_server(
        server=AsyncServer(strategy=strategy, max_workers=8, client_manager=SimpleClientManager(), async_strategy=AsynchronousStrategy(total_samples=len(trainloaders[0].dataset), alpha=0.5, staleness_alpha=0.5, fedasync_mixing_alpha=0.5, fedasync_a=0.5, num_clients=cfg.num_clients, async_aggregation_strategy="fedasync", use_staleness=True, use_sample_weighing=False)),
        server_address="0.0.0.0:8080"
    )

    ## 6. Save results

if __name__ == "__main__":
    eval: bool = False
    if not eval:
        main()
    else:
        trainloaders, validationloaders, testloader = prepare_dataset_nonIID(10, 64)
                