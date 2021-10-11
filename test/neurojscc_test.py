import pickle

import torch
import numpy as np
from snn.utils.misc import str2bool, make_network_parameters
from snn.models.SNN import BinarySNN
from snn.utils.utils_snn import refractory_period
from snn.data_preprocessing.load_data import get_example

from utils.misc import channel


def get_acc_neurojscc(encoder, decoder, n_output_enc, test_iter, n_examples_test, T, systematic, snr):
    encoder.eval()
    encoder.reset_internal_state()

    decoder.eval()
    decoder.reset_internal_state()

    outputs = torch.zeros([len(test_iter), decoder.n_output_neurons, T])

    true_labels = np.array([])

    for j in range(n_examples_test):
        sample_enc, label = next(test_iter)
        refractory_period(encoder)  # reset networks between examples
        refractory_period(decoder)

        sample_enc = sample_enc[0].to(encoder.device)
        label = torch.sum(label, dim=-1).argmax(-1)
        true_labels = np.hstack((true_labels, label.cpu().numpy()))

        for t in range(T):
            encoder(sample_enc[t])  # propagate through encoder

            ### Transmit sample through the channel
            if systematic:
                decoder_input = channel(torch.cat((sample_enc[t], encoder.spiking_history[encoder.hidden_neurons[-n_output_enc:], -1])), decoder.device, snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-n_output_enc:], -1], decoder.device, snr)

            decoder(decoder_input)  # propagate through decoder
            outputs[j, :, t] = decoder.spiking_history[decoder.output_neurons, -1]

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    accs_final = np.sum(predictions.cpu().numpy() == true_labels) / len(predictions)

    accs_pf = torch.zeros([T], dtype=torch.float) # accuracy per frame
    for t in range(1, T):
        predictions = torch.sum(outputs[:, :, :t], dim=-1).argmax(-1)
        acc = np.sum(predictions.cpu().numpy() == true_labels) / len(predictions)
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

    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for _ in range(args.num_ite):
        for snr in args.snr_list:
            args.snr = snr

            test_indices = np.random.choice(find_indices_for_labels(args.dataset.root.test, args.labels), [args.num_samples_test], replace=False)
            accs_final, accs_pf = get_acc_neurojscc(encoder, decoder, args.n_output_enc, args.dataset.root.test, test_indices, T, args.n_classes,
                                                    args.input_shape, args.dt, args.dataset.root.stats.train_data[1], args.polarity, args.systematic, snr)

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

    with open(weights + r'/acc_per_snr_final.pkl', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame.pkl', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
