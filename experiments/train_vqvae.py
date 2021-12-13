import pickle

import numpy as np
import tables
import torch
import argparse
from snn.models.SNN import BinarySNN

from utils import training_utils
from test import testing_utils
from utils.training_utils import init_vqvae
from utils.misc import get_intermediate_dims
from neurodata.load_data import create_dataloader


def train_vqvae_ldpc(args):
    # Make VAE
    args.residual = 80 % args.n_frames
    if args.residual:
        args.n_frames += 1

    input_size = [np.prod([1 + args.polarity, args.x_max, args.x_max])]
    args.input_shape = input_size
    T = int(args.sample_length * 1000 / args.dt)

    # Create dataloaders
    train_dl, test_dl = create_dataloader(args.dataset_path, batch_size=1, size=input_size, classes=[1, 7],
                                          sample_length_train=args.sample_length * 1000, sample_length_test=args.sample_length * 1000, dt=args.dt, polarity=args.polarity, ds=1,
                                          shuffle_test=True, num_workers=0)
    vqvae, vqvae_optimizer = init_vqvae(args, T)

    args.quantized_dim, args.encodings_dim = get_intermediate_dims(vqvae, args, T, train_dl)

    # Make classifier
    classifier = training_utils.init_classifier(args, T)

    # LDPC coding
    args.H, args.G, args.k = training_utils.init_ldpc(args.encodings_dim)


    # Training
    train_res_recon_error = []
    train_res_perplexity = []

    train_iter = iter(train_dl)
    for i in range(args.num_samples_train):
        try:
            example, label = next(train_iter)
        except IndexError:
            train_iter = iter(train_dl)
            example, label = next(train_iter)

        example = example[0]
        label = label[0]

        train_res_recon_error, train_res_perplexity = \
            training_utils.train_vqvae(vqvae, vqvae_optimizer, args, train_res_recon_error, train_res_perplexity, example)
        training_utils.train_classifier(classifier, args, example, label)

        if (i + 1) % args.test_period == 0:
            print('Testing at step %d...' % (i + 1))
            acc, _ = testing_utils.get_acc_classifier(classifier, vqvae, args, test_dl)
            print('test accuracy: %f' % acc)
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-args.test_period:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-args.test_period:]))

            args.test_accs[int(i + 1)].append(acc)
            with open(args.save_path + r'/test_accs.pkl', 'wb') as f:
                pickle.dump(args.test_accs, f, pickle.HIGHEST_PROTOCOL)
            if isinstance(classifier, BinarySNN):
                classifier.save(args.save_path + r'/snn_weights.hdf5')
            else:
                torch.save(classifier.state_dict(), args.save_path + r'mlp_weights.pt')
            torch.save(vqvae.state_dict(), args.save_path + r'vqvae_weights.pt')

