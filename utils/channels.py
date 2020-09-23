import torch


class Channel:
    def __init__(self):
        self.unnoisy_output = None

    def generate_noise(self, snr_db, device):
        sig_avg_db = 10 * torch.log10(torch.mean(self.unnoisy_output))
        noise_db = sig_avg_db - snr_db
        sigma_noise = 10 ** (noise_db / 10)

        noise = torch.normal(0, torch.ones(self.unnoisy_output.shape) * sigma_noise).to(device)

        return noise

    def reset(self):
        self.unnoisy_output = None


class MultiPathChannel(Channel):
    def __init__(self, dim, tau, length_filter):
        super(MultiPathChannel, self).__init__()
        self.tau = tau
        self.length_filter = length_filter
        self.exp_filter = torch.cat([torch.exp(torch.FloatTensor([- t/tau for t in range(length_filter + 1)]))[1:].unsqueeze(1) for _ in range(dim)], dim=1)

    def propagate(self, signal, device, snr_db):
        if self.unnoisy_output is None:
            self.unnoisy_output = signal.unsqueeze(0)
            noise = self.generate_noise(snr_db, device)
            channel_output = self.unnoisy_output + noise

        else:
            self.unnoisy_output = torch.cat((self.unnoisy_output, signal.unsqueeze(0)), dim=0)
            noise = self.generate_noise(snr_db, device)
            channel_output = self.unnoisy_output + noise

            for i in range(len(self.unnoisy_output) - 1):
                len_multipath = min(len(self.unnoisy_output) - i - 1, self.length_filter)
                multipath_sig = channel_output[i] * self.exp_filter[:len_multipath]
                channel_output[i + 1: i + 1 + len_multipath] = multipath_sig

        return channel_output


class RicianChannel(Channel):
    def __init__(self):
        super(RicianChannel, self).__init__()

    def propagate(self, signal, device, snr_db):
        if self.unnoisy_output is None:
            self.unnoisy_output = signal.unsqueeze(0)
        else:
            self.unnoisy_output = torch.cat((self.unnoisy_output, signal.unsqueeze(0)), dim=0)

        noise = self.generate_noise(snr_db, device)

        channel_gains = 1 + torch.normal(0, torch.ones(signal.shape) * torch.sqrt(torch.ones([1]))).to(device)
        channel_output = channel_gains * signal + noise

        return channel_output
