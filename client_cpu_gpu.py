from client import generate_client_fn_cpu_gpu
from dataset import prepare_dataset, prepare_dataset_nonIID
import argparse

import flwr as fl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--cpu', type=bool, default=True) 
    parser.add_argument('--cid', type=str)
    args = parser.parse_args()

    trainloaders, validationloaders, testloader = prepare_dataset_nonIID(10, 64)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=generate_client_fn_cpu_gpu(trainloaders, validationloaders, 10)(args.cid, args.cpu),
    )
