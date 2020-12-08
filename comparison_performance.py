import pickle

import torch
import tables
import argparse
from snn.utils.misc import str2bool, find_indices_for_labels

from test.ook import ook_test, ook_ldpc_test
from test.vqvae_ldpc import vqvae_test
from test.neurojscc_test import neurojscc_test


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home')
    parser.add_argument('--results')
    parser.add_argument('--model', choices=['neurojscc', 'ook', 'ook_ldpc', 'vqvae'])
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--num_ite', default=1, type=int)
    parser.add_argument('--snr_list', nargs='+', default=None, type=int, help='')
    parser.add_argument('--n_frames', default=80, type=int, help='')

    parser.add_argument('--dt', default=25000, type=int, help='')
    parser.add_argument('--sample_length', default=2000, type=int, help='')
    parser.add_argument('--input_shape', nargs='+', default=[676], type=int, help='Shape of an input sample')
    parser.add_argument('--polarity', default='false', type=str, help='Use polarity or not')

    parser.add_argument('--classifier', type=str, default='snn', choices=['snn', 'mlp'])
    parser.add_argument('--classifier_weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--maxiter', default=100, type=int, help='Max number of iteration for BP decoding of LDPC code')
    parser.add_argument('--ldpc_rate', default=2, type=float)

    parser.add_argument('--systematic', type=str, default='false', help='Systematic communication')
    parser.add_argument('--n_output_enc', default=128, type=int, help='')
    parser.add_argument('--n_h', default=256, type=int, help='Number of hidden neurons')

    args = parser.parse_args()

print(args)

if args.weights is not None:
    try:
        exp_args_path = args.results + args.weights + '/commandline_args.pkl'
        args_dict = vars(args)

        with open(exp_args_path, 'rb') as f:
            exp_args = pickle.load(f)

        for key in exp_args.keys():
            args_dict[key] = exp_args[key]

    except FileNotFoundError:
        pass


dataset = args.home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5'
args.dataset = tables.open_file(dataset)

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.save_path = None

args.labels = [1, 7]
args.polarity = str2bool(args.polarity)
args.T = int(args.sample_length * 1000 / args.dt)
### Learning parameters
args.num_samples_test = args.dataset.root.stats.test_data[0]
args.num_samples_test = min(args.num_samples_test, len(find_indices_for_labels(args.dataset.root.test, args.labels)))

### Network parameters
if args.polarity:
    args.n_input_neurons = int(2 * (args.dataset.root.stats.train_data[1] ** 2))
else:
    args.n_input_neurons = int(args.dataset.root.stats.train_data[1] ** 2)
args.n_output_neurons = args.dataset.root.stats.train_label[1]
args.n_hidden_neurons = args.n_h
args.n_classes = args.dataset.root.stats.test_label[1]

if args.model == 'neurojscc' or args.model == 'wispike':
    neurojscc_test(args)

elif args.model == 'ook':
    ook_test(args)

elif args.model == 'ook_ldpc':
    ook_ldpc_test(args)

elif args.model == 'vqvae':
    vqvae_test(args)
