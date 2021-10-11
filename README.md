# NeuroJSCC

Code for training end-to-end neuromorphic systems, as presented in 

N. Skatchkovsky, H. Jang, and O. Simeone, End-to-End Learning of Neuromorphic Wireless Systems for Low-Power Edge Artificial Intelligence, to be presented at Asilomar 2020
https://arxiv.org/abs/2009.01527

To run, this code requires to install our `snn` package, which can be found at https://github.com/kclip/snn
This repo will not be maintained actively, the latest working version of the snn package is commit 6ba3465
# Run example
An experiment can be run on the MNIST-DVS dataset by launching

`python snn/launch_experiment.py`

Make sure to first download and preprocess the MNIST-DVS dataset using the script in `snn/data_preprocessing/process_mnistdvs.py` and change your home directory in `snn/launch_experiment.py`.

