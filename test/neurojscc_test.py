import pickle

import torch
import numpy as np
from snn.utils.misc import str2bool, make_network_parameters, find_indices_for_labels
from snn.models.SNN import BinarySNN
from snn.utils.utils_snn import refractory_period
from snn.data_preprocessing.load_data import get_example
from utils.channels import MultiPathChannel, RicianChannel, Channel


def get_acc_neurojscc(encoder, decoder, channel, n_output_enc, hdf5_group, test_indices, T, n_classes, input_shape, dt, x_max, polarity, systematic, snr):
    encoder.eval()
    encoder.reset_internal_state()

    decoder.eval()
    decoder.reset_internal_state()

    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, T])

    true_classes = torch.LongTensor(hdf5_group.labels[test_indices, 0])
    hidden_hist = torch.zeros([encoder.n_hidden_neurons + decoder.n_hidden_neurons, T])

    for j, idx in enumerate(test_indices):
        refractory_period(encoder)
        refractory_period(decoder)
        channel.reset()

        sample_enc, _ = get_example(hdf5_group, idx, T, n_classes, input_shape, dt, x_max, polarity)
        sample_enc = sample_enc.to(encoder.device)

        for t in range(T):
            _ = encoder(sample_enc[:, t])

            if systematic:
                decoder_input = channel(torch.cat((sample_enc[:, t], encoder.spiking_history[encoder.hidden_neurons[-n_output_enc:], -1])), decoder.device, snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-n_output_enc:], -1], decoder.device, snr)

            _ = decoder(decoder_input)
            outputs[j, :, t] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, t] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, t] = decoder.spiking_history[decoder.hidden_neurons, -1]

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    accs_final = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    accs_pf = torch.zeros([T], dtype=torch.float)
    for t in range(1, T):
        predictions = torch.sum(outputs[:, :, :t], dim=-1).argmax(-1)
        acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
        accs_pf[t] = acc

    return accs_final, accs_pf


def neurojscc_test(args):
    T = int(args.sample_length * 1000 / args.dt)

    n_hidden_enc = args.n_h

    args.systematic = str2bool(args.systematic)
    if args.systematic:
        n_transmitted = args.n_input_neurons + args.n_output_enc
    else:
        n_transmitted = args.n_output_enc

    n_inputs_dec = n_transmitted
    n_hidden_dec = args.n_h

    encoder = BinarySNN(**make_network_parameters('snn', args.n_input_neurons, 0, n_hidden_enc),
                        device=args.device)

    decoder = BinarySNN(**make_network_parameters('snn', n_inputs_dec, args.n_output_neurons, n_hidden_dec),
                        device=args.device)

    weights = args.results + args.weights
    encoder_weights = weights + r'/encoder_weights_final.hdf5'
    decoder_weights = weights + r'/decoder_weights_final.hdf5'

    encoder.import_weights(encoder_weights)
    decoder.import_weights(decoder_weights)

    if args.channel_type == 'awgn':
        args.channel = Channel()
    elif args.channel_type == 'multipath':
        args.channel = MultiPathChannel(n_transmitted, args.tau_channel, args.channel_length)
    elif args.channel_type == 'rician':
        args.channel = RicianChannel()


    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for _ in range(args.num_ite):
        for snr in args.snr_list:
            args.snr = snr

            test_indices = np.random.choice(find_indices_for_labels(args.dataset.root.test, args.labels), [args.num_samples_test], replace=False)
            accs_final, accs_pf = get_acc_neurojscc(encoder, decoder, args.channel, args.n_output_enc, args.dataset.root.test, test_indices, T, args.n_classes,
                                                    args.input_shape, args.dt, args.dataset.root.stats.train_data[1], args.polarity, args.systematic, snr)

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

    with open(weights + r'/acc_per_snr_final.pkl', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame.pkl', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
