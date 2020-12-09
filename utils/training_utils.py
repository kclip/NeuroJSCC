import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pyldpc

from snn.training_utils.snn_training import feedforward_sampling, local_feedback_and_update, init_training, train_on_example
from snn.models.SNN import BinarySNN
from snn.utils.filters import get_filter
from snn.utils.misc import make_network_parameters
from snn.data_preprocessing.load_data import get_example

from utils.misc import example_to_framed
from models.mlp import MLP
from models.vqvae import Model


### VQ-VAE & LDPC
def init_vqvae(args, T):
    num_input_channels = T // args.n_frames

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    commitment_cost = 0.25

    decay = 0.99

    vqvae = Model(num_input_channels, num_hiddens, num_residual_layers, num_residual_hiddens, args.num_embeddings, args.embedding_dim, commitment_cost, decay).to(args.device)
    optimizer = optim.Adam(vqvae.parameters(), lr=args.lr_vqvae, amsgrad=False)

    return vqvae, optimizer


def train_vqvae(model, optimizer, args, train_res_recon_error, train_res_perplexity, idx):
    model.train()

    T = int(args.sample_length * 1000 / args.dt)
    example, label = get_example(args.dataset.root.train, idx, T, args.n_classes, args.input_shape, args.dt, args.dataset.root.stats.train_data[1], args.polarity)

    framed = example_to_framed(example, args, T)

    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(framed)
    recon_error = F.mse_loss(data_recon, framed)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    return train_res_recon_error, train_res_perplexity


def init_ldpc(message_dim):
    ldpc_codewords_length = int(1.5 * np.prod(message_dim))
    ldpc_codewords_length += ldpc_codewords_length % 3
    d_v = 2
    d_c = 6

    # Make LDPC
    H, G = pyldpc.make_ldpc(ldpc_codewords_length, d_v, d_c, systematic=True, sparse=True)
    n, k = G.shape

    assert k >= np.prod(message_dim)

    return H, G, k


### Classifiers
def init_classifier(args, T):
    if args.classifier == 'snn':
        n_input_neurons = args.dataset.root.stats.train_data[1] ** 2
        n_output_neurons = args.dataset.root.stats.train_label[1]


        classifier = BinarySNN(**make_network_parameters(network_type=args.model,
                                                         n_input_neurons=n_input_neurons,
                                                         n_output_neurons=n_output_neurons,
                                                         n_hidden_neurons=args.n_hidden_neurons,
                                                         topology_type=args.topology_type,
                                                         topology=None,
                                                         n_neurons_per_layer=args.n_neurons_per_layer,
                                                         density=args.density,
                                                         weights_magnitude=args.weights_magnitude,
                                                         initialization=args.initialization,
                                                         synaptic_filter=get_filter(args.syn_filter),
                                                         n_basis_ff=args.n_basis_ff,
                                                         n_basis_fb=args.n_basis_fb,
                                                         tau_ff=args.tau_ff,
                                                         tau_fb=args.tau_fb,
                                                         mu=args.mu
                                                         ),
                               device=args.device)


        args.eligibility_trace_output, args.eligibility_trace_hidden, \
            args.learning_signal, args.baseline_num, args.baseline_den = init_training(classifier)

    elif args.classifier == 'mlp':
        n_input_neurons = args.dataset.root.stats.train_data[1] * T
        n_output_neurons = args.dataset.root.stats.train_label[1]

        classifier = MLP(n_input_neurons, args.n_h, n_output_neurons)
        args.mlp_optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr)
        args.mlp_criterion = torch.nn.CrossEntropyLoss()

    else:
        raise NotImplementedError

    return classifier


def train_mlp(model, example, label, optimizer, criterion):
    # clear the gradients of all optimized variables
    optimizer.zero_grad()

    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(example)

    # calculate the loss
    loss = criterion(output.unsqueeze(0), label.unsqueeze(0))

    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()


def train_classifier(model, args, idx):
    T = int(args.sample_length * 1000 / args.dt)
    inputs, label = get_example(args.dataset.root.train, idx, T, args.n_classes, args.input_shape, args.dt, args.dataset.root.stats.train_data[1], args.polarity)

    if isinstance(model, BinarySNN):
        inputs = inputs.to(model.device)
        label = label.to(model.device)

        model.train()
        log_proba, args.eligibility_trace_hidden, args.eligibility_trace_output, args.learning_signal, args.baseline_num, args.baseline_den = \
            train_on_example(model, T, inputs, label, args.gamma, args.r, args.eligibility_trace_hidden, args.eligibility_trace_output,
                             args.learning_signal, args.baseline_num, args.baseline_den, args.lr, args.beta, args.kappa)


    elif isinstance(model, MLP):
        example = torch.FloatTensor(args.dataset.root.train.data[idx]).flatten()
        label = torch.tensor(np.argmax(np.sum(label, axis=(-1)), axis=-1))

        train_mlp(model, example, label, args.mlp_optimizer, args.mlp_criterion)


### WiSpike
def init_training_wispike(encoder, decoder, args):
    encoder.train()
    decoder.train()

    eligibility_trace_hidden_enc = {parameter: encoder.get_gradients()[parameter]for parameter in encoder.get_gradients()}
    eligibility_trace_hidden_dec = {parameter: decoder.get_gradients()[parameter][decoder.hidden_neurons - decoder.n_input_neurons] for parameter in decoder.get_gradients()}
    eligibility_trace_output_dec = {parameter: decoder.get_gradients()[parameter][decoder.output_neurons - decoder.n_input_neurons] for parameter in decoder.get_gradients()}

    learning_signal = 0

    baseline_num_enc = {parameter: eligibility_trace_hidden_enc[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden_enc}
    baseline_den_enc = {parameter: eligibility_trace_hidden_enc[parameter].pow(2) for parameter in eligibility_trace_hidden_enc}

    baseline_num_dec = {parameter: eligibility_trace_hidden_dec[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden_dec}
    baseline_den_dec = {parameter: eligibility_trace_hidden_dec[parameter].pow(2) for parameter in eligibility_trace_hidden_dec}

    T = args.dataset.root.stats.train_data[-1]

    return eligibility_trace_hidden_enc, eligibility_trace_hidden_dec, eligibility_trace_output_dec,\
           learning_signal, baseline_num_enc, baseline_den_enc, baseline_num_dec, baseline_den_dec, T

