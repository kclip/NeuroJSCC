from __future__ import print_function
import pickle

import argparse
import numpy as np
from snn.utils.misc import mksavedir, str2bool
import tables
import torch

from experiments.neuroJSCC import train_neurojscc
from experiments.train_vqvae import train_vqvae_ldpc

''''
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset', default='mnist_dvs')
    parser.add_argument('--model', default='neurojscc', choices=['neurojscc', 'vqvae'])
    parser.add_argument('--num_ite', default=5, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--test_period', default=1000, type=int, help='')

    parser.add_argument('--dt', default=25000, type=int, help='')
    parser.add_argument('--sample_length', default=2000, type=int, help='')
    parser.add_argument('--input_shape', nargs='+', default=[676], type=int, help='Shape of an input sample')
    parser.add_argument('--polarity', default='false', type=str, help='Use polarity or not')

    parser.add_argument('--num_samples_train', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--start_idx', type=int, default=0, help='When resuming training from existing weights, index to start over from')
    parser.add_argument('--labels', nargs='+', default=None, type=int, help='Class labels to be used during training')

    parser.add_argument('--save_path', type=str, default=None, help='')
    parser.add_argument('--suffix', type=str, default='', help='Appended to the name of the saved results and weights')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')


    # Arguments common to all models
    parser.add_argument('--n_h', default=256, type=int, help='Number of hidden neurons')
    parser.add_argument('--topology_type', default='fully_connected', type=str, choices=['fully_connected', 'feedforward', 'layered', 'custom'], help='Topology of the network')
    parser.add_argument('--density', default=None, type=int, help='Density of the connections if topology_type is "sparse"')
    parser.add_argument('--n_neurons_per_layer', default=0, type=int, help='Number of neurons per layer if topology_type is "layered"')
    parser.add_argument('--initialization', default='uniform', type=str, choices=['uniform', 'glorot'], help='Initialization of the weights')
    parser.add_argument('--weights_magnitude', default=0.05, type=float, help='Magnitude of weights at initialization')

    parser.add_argument('--n_basis_ff', default=8, type=int, help='Number of basis functions for synaptic connections')
    parser.add_argument('--syn_filter', default='raised_cosine_pillow_08', type=str,
                        choices=['base_filter', 'cosine_basis', 'raised_cosine', 'raised_cosine_pillow_05', 'raised_cosine_pillow_08'],
                        help='Basis function to use for synaptic connections')
    parser.add_argument('--tau_ff', default=10, type=int, help='Feedforward connections time constant')
    parser.add_argument('--n_basis_fb', default=1, type=int, help='Number of basis functions for feedback connections')
    parser.add_argument('--tau_fb', default=10, type=int, help='Feedback connections time constant')
    parser.add_argument('--mu', default=1.5, type=float, help='Width of basis functions')

    parser.add_argument('--kappa', default=0.2, type=float, help='eligibility trace decay coefficient')
    parser.add_argument('--r', default=0.3, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization strength')


    # Arguments for communications
    parser.add_argument('--rand_snr', type=str, default='false', help='Use random SNR for each sample during training to make it more robust')
    parser.add_argument('--snr', type=float, default=0, help='SNR')

    # Argument for NeuroJSCC
    parser.add_argument('--systematic', type=str, default='true', help='Systematic communication')
    parser.add_argument('--n_output_enc', default=128, type=int, help='Number of the hidden neurons that are output neurons for the encoder')


    # Arguments for VQVAE + LDPC
    parser.add_argument('--classifier', type=str, default='snn', choices=['snn', 'mlp'])
    parser.add_argument('--embedding_dim', default=32, type=int, help='Size of VQ-VAE latent embeddings')
    parser.add_argument('--num_embeddings', default=10, type=int, help='Number of VQ-VAE latent embeddings')
    parser.add_argument('--lr_vqvae', default=1e-3, type=float, help='Learning rate of VQ-VAE')
    parser.add_argument('--maxiter', default=100, type=int, help='Max number of iteration for BP decoding of LDPC code')
    parser.add_argument('--n_frames', default=80, type=int, help='')


    args = parser.parse_args()

print(args)

if args.where == 'local':
    home = r'C:/Users/K1804053/PycharmProjects'
elif args.where == 'rosalind':
    home = r'/users/k1804053'
elif args.where == 'jade':
    home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    home = r'/home/k1804053'


datasets = {'mnist_dvs': r'mnist_dvs_events.hdf5',
            'dvs_gesture': r'dvs_gestures_events.hdf5'
            }


if args.dataset[:3] == 'shd':
    dataset = home + r'/datasets/shd/' + datasets[args.dataset]
elif args.dataset[:5] == 'mnist':
    dataset = home + r'/datasets/mnist-dvs/' + datasets[args.dataset]
elif args.dataset[:11] == 'dvs_gesture':
    dataset = home + r'/datasets/DvsGesture/' + datasets[args.dataset]
elif args.dataset[:7] == 'swedish':
    dataset = home + r'/datasets/SwedishLeaf_processed/' + datasets[args.dataset]
else:
    print('Error: dataset not found')

dataset = tables.open_file(dataset)
args.polarity = str2bool(args.polarity)
args.n_classes = dataset.root.stats.test_label[1]

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

### Learning parameters
if not args.num_samples_train:
    args.num_samples_train = dataset.root.stats.train_data[0]

if args.test_period is not None:
    if not args.num_samples_test:
        args.num_samples_test = dataset.root.stats.test_data[0]
    args.ite_test = np.arange(0, args.num_samples_train, args.test_period)

    if args.save_path is not None:
        with open(args.save_path + '/test_accs.pkl', 'rb') as f:
            args.test_accs = pickle.load(f)
    else:
        args.test_accs = {i: [] for i in args.ite_test}
        args.test_accs[args.num_samples_train] = []


# Save results and weights
if args.model == 'neurojscc':
    name = args.dataset + r'_' + args.model + r'_%d_epochs_nh_%d_nout_%d' % (args.num_samples_train, args.n_h, args.n_output_enc) + args.suffix
else:
    name = 'vqvae_' + args.classifier + r'_%d_epochs_nh_%d_nemb_%d_nframes_%d' % (args.num_samples_train, args.n_h, args.num_embeddings, args.n_frames) + args.suffix

results_path = home + r'/results/'
if args.save_path is None:
    args.save_path = mksavedir(pre=results_path, exp_dir=name)

with open(args.save_path + 'commandline_args.pkl', 'wb') as f:
    pickle.dump(args.__dict__, f, pickle.HIGHEST_PROTOCOL)


args.dataset = dataset

### Network parameters
if args.polarity:
    args.n_input_neurons = int(2 * (args.dataset.root.stats.train_data[1] ** 2))
else:
    args.n_input_neurons = int(args.dataset.root.stats.train_data[1] ** 2)
args.n_output_neurons = args.dataset.root.stats.train_label[1]
args.n_hidden_neurons = args.n_h


if args.topology_type == 'custom':
    args.topology = torch.zeros([args.n_hidden_neurons + args.n_output_neurons,
                                 args.n_input_neurons + args.n_hidden_neurons + args.n_output_neurons])
    args.topology[-args.n_output_neurons:, args.n_input_neurons:-args.n_output_neurons] = 1
    args.topology[:args.n_hidden_neurons, :(args.n_input_neurons + args.n_hidden_neurons)] = 1
    # Feel free to fill this with any custom topology
    print(args.topology)
else:
    args.topology = None

args.rand_snr = str2bool(args.rand_snr)

# Training
if args.model == 'neurojscc':
    train_neurojscc(args)

elif args.model == 'vqvae':
    train_vqvae_ldpc(args)


