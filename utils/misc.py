import torch
import numpy as np
import pyldpc

from snn.data_preprocessing.load_data import get_example


def binarize(signal):
    signal[signal >= 0.5] = 1.
    signal[signal < 0.5] = 0.
    return signal


def channel(signal, device, snr_db):
    sig_avg_db = 10 * torch.log10(torch.mean(signal))
    noise_db = sig_avg_db - snr_db
    sigma_noise = 10 ** (noise_db / 10)

    noise = torch.normal(0, torch.ones(signal.shape) * sigma_noise)
    channel_output = signal + noise.to(device)

    channel_output = binarize(channel_output)
    return channel_output


def channel_coding_decoding(args, message):
    # Transmit through channel
    message_shape = message.shape

    to_send = np.zeros([args.k])
    to_send[:np.prod(message_shape)] = message.flatten()

    coded_quantized = pyldpc.encode(args.G, to_send, args.snr)
    received = pyldpc.decode(args.H, coded_quantized, args.snr, args.maxiter)

    decoded = received[:np.prod(message_shape)]
    decoded = decoded.reshape(*message_shape)

    return decoded


def example_to_framed(example, args, T):
    if args.residual:
        frames = torch.zeros([args.n_frames, T // (args.n_frames - 1), args.dataset.root.stats.train_data[1], args.dataset.root.stats.train_data[1]])
        frames[:-1] = example[:, :(args.n_frames - 1) * (T // (args.n_frames - 1))].transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape(frames[:-1].shape)
        frames[-1, :args.residual] = example[:, -args.residual:].transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([args.residual, args.dataset.root.stats.train_data[1], args.dataset.root.stats.train_data[1]])
    else:
        frames = torch.FloatTensor(example).transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([args.n_frames, T // args.n_frames, args.dataset.root.stats.train_data[1], args.dataset.root.stats.train_data[1]])

    return frames


def framed_to_example(frames, args, T):
    if args.residual:
        data_reconstructed = torch.zeros([args.dataset.root.stats.train_data[1] ** 2, T])
        data_reconstructed[:, :(args.n_frames - 1) * (T // (args.n_frames - 1))] \
            = frames[:-1].reshape([(args.n_frames - 1) * (T // (args.n_frames - 1)), -1]).transpose(1, 0)
        data_reconstructed[:, -args.residual:] = frames[-1, :args.residual].reshape([args.residual, -1]).transpose(1, 0)
    else:
        data_reconstructed = frames.reshape([-1, args.dataset.root.stats.train_data[1] ** 2]).transpose(1, 0)

    return data_reconstructed


def get_intermediate_dims(vqvae, args, T, dataset):
    example, _ = get_example(dataset.root.train, 0, T, args.n_classes, args.input_shape, args.dt, dataset.root.stats.train_data[1], args.polarity)
    example_frame = example_to_framed(example, args, T)[0].unsqueeze(0)
    args.frame_shape = example_frame.shape

    example_quantized, example_encodings = vqvae.encode(example_frame)
    encodings_dim = example_encodings.data.numpy().shape
    quantized_dim = example_quantized.data.clone().permute(0, 2, 3, 1).contiguous().shape

    return quantized_dim, encodings_dim
