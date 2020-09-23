import pickle
import os

import torch
import numpy as np

from snn.utils.utils_snn import refractory_period
from snn.utils.misc import get_indices, make_network_parameters, str2bool
from snn.models.SNN import BinarySNN
from snn.training_utils.snn_training import local_feedback_and_update
from snn.utils.filters import get_filter
from snn.data_preprocessing.load_data import get_example

from utils.training_utils import init_training_wispike
from test.neurojscc_test import get_acc_neurojscc
from utils.channels import MultiPathChannel, RicianChannel, Channel


def train_neurojscc(args):
    ### Network parameters
    n_hidden_enc = args.n_h

    args.systematic = str2bool(args.systematic)
    if args.systematic:
        n_transmitted = args.n_input_neurons + args.n_output_enc
    else:
        n_transmitted = args.n_output_enc

    n_inputs_dec = n_transmitted
    n_hidden_dec = args.n_h

    if args.channel_type == 'awgn':
        args.channel = Channel()
    elif args.channel_type == 'multipath':
        args.channel = MultiPathChannel(n_transmitted, args.tau_channel, args.channel_length)
    elif args.channel_type == 'rician':
        args.channel = RicianChannel()

    args.lr = args.lr / (n_hidden_enc + n_hidden_dec)

    for _ in range(args.num_ite):
        ### Find indices
        indices, test_indices = get_indices(args)

        encoder = BinarySNN(**make_network_parameters(network_type=args.model,
                                                      n_input_neurons=args.n_input_neurons,
                                                      n_output_neurons=0,
                                                      n_hidden_neurons=n_hidden_enc,
                                                      topology_type=args.topology_type,
                                                      topology=args.topology,
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

        decoder = BinarySNN(**make_network_parameters(network_type=args.model,
                                                      n_input_neurons=n_inputs_dec,
                                                      n_output_neurons=args.n_output_neurons,
                                                      n_hidden_neurons=n_hidden_dec,
                                                      topology_type=args.topology_type,
                                                      topology=args.topology,
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


        if args.start_idx > 0:
            encoder.import_weights(args.save_path + r'encoder_weights.hdf5')
            decoder.import_weights(args.save_path + r'decoder_weights.hdf5')

        train_data = args.dataset.root.train
        test_data = args.dataset.root.test
        T = int(args.sample_length * 1000 / args.dt)

        # init training
        eligibility_trace_hidden_enc, eligibility_trace_hidden_dec, eligibility_trace_output_dec, \
            learning_signal, baseline_num_enc, baseline_den_enc, baseline_num_dec, baseline_den_dec, S_prime = init_training_wispike(encoder, decoder, args)

        for j, idx in enumerate(indices):

            if args.test_accs:
                if (j + 1) in args.test_accs:
                    acc, _ = get_acc_neurojscc(encoder, decoder, args.channel, args.n_output_enc, test_data, test_indices, T, args.n_classes, args.input_shape,
                                               args.dt, args.dataset.root.stats.train_data[1], args.polarity, args.systematic, args.snr)
                    print('test accuracy at ite %d: %f' % (int(j + 1), acc))
                    args.test_accs[int(j + 1)].append(acc)

                    if args.save_path is not None:
                        with open(args.save_path + '/test_accs.pkl', 'wb') as f:
                            pickle.dump(args.test_accs, f, pickle.HIGHEST_PROTOCOL)

                        encoder.save(args.save_path + '/encoder_weights.hdf5')
                        decoder.save(args.save_path + '/decoder_weights.hdf5')

                    encoder.train()
                    decoder.train()

            refractory_period(encoder)
            refractory_period(decoder)
            args.channel.reset()

            sample_enc, output_dec = get_example(train_data, idx, T, args.n_classes, args.input_shape, args.dt, args.dataset.root.stats.train_data[1], args.polarity)

            if args.rand_snr:
                args.snr = np.random.choice(np.arange(0, -9, -1))

            for t in range(T):
                # Feedforward sampling encoder
                log_proba_enc = encoder(sample_enc[:, t])
                proba_hidden_enc = torch.sigmoid(encoder.potential[encoder.hidden_neurons - encoder.n_input_neurons])

                if args.systematic:
                    decoder_input = args.channel.propagate(torch.cat((sample_enc[:, t],
                                                                      encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1])),
                                                           decoder.device, args.snr)
                else:
                    decoder_input = args.channel.propagate(encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1],
                                                           decoder.device, args.snr)

                sample_dec = torch.cat((decoder_input, output_dec[:, t]), dim=0).to(decoder.device)

                log_proba_dec = decoder(sample_dec)
                proba_hidden_dec = torch.sigmoid(decoder.potential[decoder.hidden_neurons - decoder.n_input_neurons])


                ls = torch.sum(log_proba_dec[decoder.output_neurons - decoder.n_input_neurons]) \
                     - args.gamma * torch.sum(torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1]))
                                              * torch.log(1e-12 + torch.cat((proba_hidden_enc, proba_hidden_dec)) / args.r)
                                              + (1 - torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1])))
                                              * torch.log(1e-12 + (1. - torch.cat((proba_hidden_enc, proba_hidden_dec))) / (1 - args.r)))

                # Local feedback and update
                eligibility_trace_hidden_dec, eligibility_trace_output_dec, learning_signal, baseline_num_dec, baseline_den_dec \
                    = local_feedback_and_update(decoder, ls, eligibility_trace_hidden_dec, eligibility_trace_output_dec,
                                                learning_signal, baseline_num_dec, baseline_den_dec, args.lr, args.beta, args.kappa)

                eligibility_trace_hidden_enc, _, _, baseline_num_enc, baseline_den_enc \
                    = local_feedback_and_update(encoder, 0, eligibility_trace_hidden_enc, None,
                                                learning_signal, baseline_num_enc, baseline_den_enc, args.lr, args.beta, args.kappa)

            if j % max(1, int(len(indices) / 5)) == 0:
                print('Step %d out of %d' % (j, len(indices)))

        # At the end of training, save final weights if none exist or if this ite was better than all the others
        if not os.path.exists(args.save_path + '/encoder_weights_final.hdf5'):
            encoder.save(args.save_path + '/encoder_weights_final.hdf5')
            decoder.save(args.save_path + '/decoder_weights_final.hdf5')
        else:
            if args.test_accs[list(args.test_accs.keys())[-1]][-1] >= max(args.test_accs[list(args.test_accs.keys())[-1]][:-1]):
                encoder.save(args.save_path + '/encoder_weights_final.hdf5')
                decoder.save(args.save_path + '/decoder_weights_final.hdf5')

