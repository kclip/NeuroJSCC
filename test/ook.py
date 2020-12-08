import pickle

import numpy as np
import pyldpc
import torch

from snn.models.SNN import BinarySNN
from snn.utils.utils_snn import *
from snn.utils.misc import make_network_parameters, find_indices_for_labels
from utils.misc import channel
from test.testing_utils import classify
from utils.misc import example_to_framed, channel_coding_decoding


def ook_test(args):
    args.residual = 0  # Samples are sent 1 by 1

    network = BinarySNN(**make_network_parameters('snn', args.n_input_neurons, args.n_output_neurons, args.n_h),
                        device=args.device)

    weights = args.results + args.classifier_weights

    network.import_weights(weights + r'/network_weights.hdf5')

    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for snr in args.snr_list:
        for _ in range(args.num_ite):
            test_indices = np.random.choice(find_indices_for_labels(args.dataset.root.test, args.labels), [args.num_samples_test], replace=False)

            predictions_final = torch.zeros([args.num_samples_test], dtype=torch.long)
            predictions_pf = torch.zeros([args.num_samples_test, args.n_frames], dtype=torch.long)
            true_classes = torch.zeros([len(test_indices)])

            for i, idx in enumerate(test_indices):
                inputs, labels = get_example(args.dataset.root.test, idx, args.T, [i for i in range(10)], args.input_shape, args.dt, 26, args.polarity)
                sample = channel(inputs.to(network.device), network.device, snr)
                predictions_final[i], predictions_pf[i] = classify(network, sample, args)

                true_classes[i] = torch.sum(labels, dim=-1).argmax(-1).type_as(true_classes)


            accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
            accs_pf = torch.zeros([args.n_frames], dtype=torch.float)

            for i in range(args.n_frames):
                acc = float(torch.sum(predictions_pf[:, i] == true_classes, dtype=torch.float) / len(predictions_pf))
                accs_pf[i] = acc

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

    with open(weights + r'/acc_per_snr_final_ook.pkl', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame_ook.pkl', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)


def ook_ldpc_test(args):
    args.n_frames = 80  # Samples are sent 1 by 1
    args.residual = 0

    network = BinarySNN(**make_network_parameters('snn', args.n_input_neurons, args.n_output_neurons, args.n_h),
                        device=args.device)

    weights = args.results + args.classifier_weights
    network.import_weights(weights + r'/network_weights.hdf5')

    example_frame = example_to_framed(args.dataset.root.train.data[0], args)[0].unsqueeze(0)
    frame_shape = example_frame.shape
    ldpc_codewords_length = int(args.ldpc_rate * np.prod(frame_shape))

    if args.ldpc_rate == 1.5:
        d_v = 2
        d_c = 6
    elif args.ldpc_rate == 2:
        d_v = 2
        d_c = 4
    elif args.ldpc_rate == 3:
        d_v = 2
        d_c = 3
    elif args.ldpc_rate == 4:
        d_v = 3
        d_c = 4
    elif args.ldpc_rate == 5:
        d_v = 4
        d_c = 5

    ldpc_codewords_length += d_c - (ldpc_codewords_length % d_c)

    # Make LDPC
    args.H, args.G = pyldpc.make_ldpc(ldpc_codewords_length, d_v, d_c, systematic=True, sparse=True)
    args.n, args.k = args.G.shape

    assert args.k >= np.prod(frame_shape)

    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for snr in args.snr_list:
        args.snr = snr
        for _ in range(args.num_ite):
            test_indices = np.random.choice(find_indices_for_labels(args.dataset.root.test, args.labels), [args.num_samples_test], replace=False)

            predictions_final = torch.zeros([args.num_samples_test], dtype=torch.long)
            predictions_pf = torch.zeros([args.num_samples_test, args.n_frames], dtype=torch.long)

            for i, idx in enumerate(test_indices):
                data = example_to_framed(args.dataset.root.test.data[idx, :, :], args, args.T)
                data_reconstructed = torch.zeros(data.shape)

                for j in range(args.n_frames):
                    frame = data[j].unsqueeze(0)
                    data_reconstructed[j] = torch.FloatTensor(channel_coding_decoding(args, frame))

                predictions_final[i], predictions_pf[i] = classify(network, data_reconstructed, args)

            true_classes = torch.LongTensor(args.dataset.root.test.labels[test_indices, 0])

            accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
            accs_pf = torch.zeros([args.n_frames], dtype=torch.float)

            for i in range(args.n_frames):
                acc = float(torch.sum(predictions_pf[:, i] == true_classes, dtype=torch.float) / len(predictions_pf))
                accs_pf[i] = acc

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

    with open(weights + r'/acc_per_snr_final_ook_ldpc_r_%3f.pkl' % args.ldpc_rate, 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame_ook_ldpc_r_%3f.pkl' % args.ldpc_rate, 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
