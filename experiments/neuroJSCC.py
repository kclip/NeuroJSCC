import pickle
import os

import torch
import numpy as np

from snn.utils.utils_snn import refractory_period
from snn.utils.misc import make_network_parameters, str2bool
from snn.models.SNN import BinarySNN
from snn.training_utils.snn_training import local_feedback_and_update
from snn.utils.filters import get_filter

from utils.misc import channel
from utils.training_utils import init_training_wispike
from test.neurojscc_test import get_acc_neurojscc
from neurodata.load_data import create_dataloader


def train_neurojscc(args):
    ### Network parameters
    n_hidden_enc = args.n_h

    # If systematic, the input signal is transmitted as well as the network outputs
    args.systematic = str2bool(args.systematic)
    if args.systematic:
        n_transmitted = args.n_input_neurons + args.n_output_enc
    else:
        n_transmitted = args.n_output_enc

    n_inputs_dec = n_transmitted
    n_hidden_dec = args.n_h

    args.lr = args.lr / (n_hidden_enc + n_hidden_dec)

    for _ in range(args.num_ite):
        ### Make networks
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

        ### If resuming training
        if args.start_idx > 0:
            encoder.import_weights(args.save_path + r'encoder_weights.hdf5')
            decoder.import_weights(args.save_path + r'decoder_weights.hdf5')

        input_size = [np.prod([1 + args.polarity, args.x_max, args.x_max])]
        T = int(args.sample_length * 1000 / args.dt)

        # Create dataloaders
        train_dl, test_dl = create_dataloader(args.dataset_path, batch_size=1, size=input_size, classes=[1, 7],
                                              sample_length_train=args.sample_length * 1000, sample_length_test=args.sample_length * 1000, dt=args.dt, polarity=args.polarity, ds=1,
                                              shuffle_test=True, num_workers=0)

        # init training
        eligibility_trace_hidden_enc, eligibility_trace_hidden_dec, eligibility_trace_output_dec, \
            learning_signal, baseline_num_enc, baseline_den_enc, baseline_num_dec, baseline_den_dec = init_training_wispike(encoder, decoder, args)

        train_iterator = iter(train_dl)
        test_iter = iter(test_dl)

        for j in range(args.num_samples_train):
            if args.test_accs:
                ### Test accuracy
                if (j + 1) in args.test_accs:
                    acc, _ = get_acc_neurojscc(encoder, decoder, args.n_output_enc, test_iter, args.num_samples_test, T, args.systematic, args.snr)
                    print('test accuracy at ite %d: %f' % (int(j + 1), acc))
                    args.test_accs[int(j + 1)].append(acc)

                    if args.save_path is not None:
                        with open(args.save_path + '/test_accs.pkl', 'wb') as f:
                            pickle.dump(args.test_accs, f, pickle.HIGHEST_PROTOCOL)

                        encoder.save(args.save_path + '/encoder_weights.hdf5')
                        decoder.save(args.save_path + '/decoder_weights.hdf5')

                    encoder.train()
                    decoder.train()

            refractory_period(encoder)  # reset snns between examples
            refractory_period(decoder)

            sample_enc, output_dec = next(train_iterator)  # input and target output
            sample_enc = sample_enc[0].to(args.device)
            output_dec = output_dec[0].to(args.device)

            if args.rand_snr:
                args.snr = np.random.choice(np.arange(0, -9, -1))

            for t in range(T):
                # Feedforward sampling encoder
                encoder(sample_enc[t])
                proba_hidden_enc = torch.sigmoid(encoder.potential[encoder.hidden_neurons - encoder.n_input_neurons])

                ### Transmit sample through the channel
                if args.systematic:
                    decoder_input = channel(torch.cat((sample_enc[t], encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1])), decoder.device, args.snr)
                else:
                    decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1], decoder.device, args.snr)

                log_proba_dec = decoder(decoder_input, output_dec[:, t])  # decoder log probability
                proba_hidden_dec = torch.sigmoid(decoder.potential[decoder.hidden_neurons - decoder.n_input_neurons])  # decoder hidden layers probabilities (for regularization)

                # compute learning signal
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

            if j % max(1, int(args.num_samples_train / 5)) == 0:
                print('Step %d out of %d' % (j, args.num_samples_train))

        # At the end of training, save final weights if none exist or if this ite was better than all the others
        if not os.path.exists(args.save_path + '/encoder_weights_final.hdf5'):
            encoder.save(args.save_path + '/encoder_weights_final.hdf5')
            decoder.save(args.save_path + '/decoder_weights_final.hdf5')
        else:
            if args.test_accs[list(args.test_accs.keys())[-1]][-1] >= max(args.test_accs[list(args.test_accs.keys())[-1]][:-1]):
                encoder.save(args.save_path + '/encoder_weights_final.hdf5')
                decoder.save(args.save_path + '/decoder_weights_final.hdf5')

