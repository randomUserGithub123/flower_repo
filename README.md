# Setup Conda Environment
 - Install `conda` for your OS:
   - [Open the link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
 - `conda create --name myenv --file spec-file.txt`
 - `conda activate myenv`

# Running Simulation
 - `conda activate myenv`
 - Run `python main.py`
 - For config: in `async_server.py` configure line `_handle_finished_future_after_fit` for logging and updating the global model and learning rate
 - Change dataset: in `main.py` change line 20 `prepare_dataset_nonIID` to be of `nonIID` or `IID` data distribution
  
# Running CPU vs GPU on different terminals (not using Flower's simulation engine)
 - `conda activate myenv`
 - Run `python server_cpu_gpu.py`
 - For CPU: run on N terminals: `python client_cpu_gpu.py --cpu=True --cid=i` where replace i from the set of `{0, 1, 2}` if N = 3
 - For GPU: run on N terminals: `python client_cpu_gpu.py --cpu=False --cid=i` where replace i from the set of `{3, 4, 5}` if N = 3
  
# Plotting Charts
 - `conda activate myenv`
 - Run `python plot_script.py`