import torch

from snn.models.SNN import BinarySNN
from snn.utils.utils_snn import *
from snn.data_preprocessing.load_data import get_example

from models.mlp import MLP
from utils.misc import channel_coding_decoding, framed_to_example, example_to_framed, binarize


def classify(classifier, example, args):
    if isinstance(classifier, BinarySNN):
        T = int(args.sample_length * 1000 / args.dt)
        if example.shape != torch.Size(args.input_shape + [T]):
            example = framed_to_example(example, args, T)
        # SNNs only accept binary inputs
        example = binarize(example)

        predictions_final, predictions_pf = classify_snn(classifier, example, args)

    elif isinstance(classifier, MLP):
        predictions_final, predictions_pf = classify_mlp(classifier, example, args)

    return predictions_final, predictions_pf


def classify_snn(network, example, args):
    network.eval()
    network.reset_internal_state()

    T = int(args.sample_length * 1000 / args.dt)
    outputs = torch.zeros([network.n_output_neurons, T])

    for t in range(T):
        _ = network(example[:, t])
        outputs[:, t] = network.spiking_history[network.output_neurons, -1]

    predictions_final = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    predictions_pf = torch.zeros([T])

    for t in range(T):
        predictions_pf[t] = torch.max(torch.sum(outputs[:, :t], dim=-1), dim=-1).indices

    return predictions_final, predictions_pf


def classify_mlp(network, example, args):
    inputs = framed_to_example(example, args, args.T).flatten()
    output = network(inputs)
    predictions_final = torch.argmax(output)
    example_padded = torch.zeros(example.shape)
    predictions_pf = torch.zeros([args.n_frames])

    for f in range(example.shape[0]):
        example_padded[f] = example[f]
        inputs = framed_to_example(example_padded, args, args.T).flatten()
        output = network(inputs)
        predictions_pf[f] = torch.argmax(output)

    return predictions_final, predictions_pf


def get_acc_classifier(classifier, vqvae, args, test_dl):
    vqvae.eval()

    test_iter = iter(test_dl)
    n_examples = len(list(test_iter))
    T = int(args.sample_length * 1000 / args.dt)
    predictions_final = torch.zeros([n_examples], dtype=torch.long)

    if args.classifier == 'snn':
        predictions_pf = torch.zeros([n_examples, T], dtype=torch.long)
    else:
        predictions_pf = torch.zeros([n_examples, args.n_frames], dtype=torch.long)

    true_classes = torch.Tensor()

    test_iter = iter(test_dl)
    for i, (example, outputs) in enumerate(test_iter):
        example = example[0]
        outputs = outputs

        data = example_to_framed(example, args, T)
        data_reconstructed = torch.zeros(data.shape)

        for f in range(args.n_frames):
            frame = data[f].unsqueeze(0)

            with torch.autograd.no_grad():
                _, encodings = vqvae.encode(frame)

                encodings_decoded = torch.FloatTensor(channel_coding_decoding(args, encodings))

                data_reconstructed[f] = vqvae.decode(encodings_decoded, args.quantized_dim)

        predictions_final[i], predictions_pf[i] = classify(classifier, data_reconstructed, args)
        true_classes = torch.cat((true_classes, torch.sum(outputs, dim=1).argmax(-1).type_as(true_classes)))

    accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
    accs_pf = torch.zeros(predictions_pf.shape, dtype=torch.float)

    if args.classifier == 'snn':
        F = T
    else:
        F = args.n_frames

    for f in range(F):
        acc = float(torch.sum(predictions_pf[:, f] == true_classes, dtype=torch.float) / len(predictions_pf))
        accs_pf[:, f] = acc

    return accs_final, accs_pf
