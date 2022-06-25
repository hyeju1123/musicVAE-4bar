import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1, bidirectional=True):

        super(Encoder, self).__init__()

        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1

        self.hidden_size = hidden_size
        self.num_hidden = num_directions * num_layers
        self.final_size = self.num_hidden * hidden_size
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional)

        self.mu = nn.Linear(self.final_size, latent_dim)
        self.std = nn.Linear(self.final_size, latent_dim)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def encode(self, x):     # (batch, seq, feat)

        x, (h, c) = self.lstm(x)
        h = h.transpose(0, 1).reshape(-1, self.final_size)

        mu = self.norm(self.mu(h))          # (batch, latent_dim)
        std = nn.Softplus()(self.std(h))    # (batch, latent_dim)

        eps = torch.randn_like(std)
        z = mu + (eps * std)                # (batch, latent_dim)
        # z = self.reparameterize(mu, std)    # (batch, latent_dim)

        return z, mu, std

    # def reparameterize(self, mu, std):
    #
    #     eps = torch.randn_like(std)
    #
    #     return mu + (eps * std)

    def forward(self, x):       # (batch, seq, feat)

        z, mu, std = self.encode(x)

        return z, mu, std


class Conductor(nn.Module):

    def __init__(self, input_size, hidden_size, device, num_layers=2, bidirectional=False):

        super(Conductor, self).__init__()

        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = num_directions * num_layers
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.conductor = nn.LSTM(batch_first=True,
                                 input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional)

    def init_hidden(self, batch_size, z):
        h0 = z.repeat(self.num_hidden, 1, 1)        # unidirectional * 2layer
        c0 = z.repeat(self.num_hidden, 1, 1)        # unidirectional * 2layer

        return h0, c0

    def forward(self, z):     # (batch, input_size) : latent space

        batch_size = z.shape[0]

        h, c = self.init_hidden(batch_size, z)
        z = z.unsqueeze(1)

        feat = torch.zeros(batch_size, 4, self.hidden_size, device=self.device)

        z_input = z
        for i in range(4):    # 4마디
            z_input, (h, c) = self.conductor(z_input, (h, c))
            feat[:, i, :] = z_input.squeeze()
            z_input = z

        feat = self.linear(feat)    # (batch, 4, hidden_size)

        return feat


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=False):

        super(Decoder, self).__init__()

        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden = num_directions * num_layers
        self.logits = nn.Linear(hidden_size, output_size)
        self.decoder = nn.LSTM(batch_first=True,
                               input_size=input_size + output_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional)

    def forward(self, x, h, c, z):

        # x: input sequence (batch, 1, feat)
        # h: LSTM state (2, batch, hidden_size)
        # c: LSTM cell (2, batch, hidden_size)
        # z: concat feature

        x = torch.cat((x, z.unsqueeze(1)), 2)

        x, (h, c) = self.decoder(x, (h, c))
        logits = self.logits(x)
        prob = nn.Softmax(dim=2)(logits)     # (batch, 1, output_size)
        out = torch.argmax(prob, 2)          # (batch, 1, output_size)

        return out, prob, h, c