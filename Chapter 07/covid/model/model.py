import copy

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)

    def forward(self, x):
        flat = x.view(x.shape[0], x.shape[1], self.input_size)
        out, h = self.gru(flat)
        return out, h


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_dl_size, output_size = 1, num_layers = 1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.lin1 = nn.Linear(hidden_size, hidden_dl_size)
        self.lin2 = nn.Linear(hidden_dl_size, output_size)

    def forward(self, x, h):
        out, h = self.gru(x.unsqueeze(0), h)
        y = torch.relu(self.lin1(out.squeeze(0)))
        y = self.lin2(y)
        return y, h


class EncoderDecoder(nn.Module):

    def __init__(self, hidden_size, hidden_dl_size, input_size = 1, output_size = 1):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = Decoder(input_size = output_size, hidden_size = hidden_size,
                               hidden_dl_size = hidden_dl_size, output_size = output_size)

    def train_model(
            self, train, target, val, val_target,
            epochs, target_len, method = 'recursive',
            tfr = 0.5, lr = 0.01, dynamic_tf = False
    ):
        losses = np.full(epochs, np.nan)
        optimizer = optim.Adam(self.parameters(), lr = lr)
        criterion = nn.MSELoss()

        best_val_loss = 1000_000
        best_model_params = None

        for e in range(epochs):
            predicted = torch.zeros(target_len, train.shape[1], 1)
            optimizer.zero_grad()
            _, enc_h = self.encoder(train)

            dec_in = train[-1, :, 0].unsqueeze(1)
            dec_h = enc_h

            if method == 'recursive':
                for t in range(target_len):
                    dec_out, dec_h = self.decoder(dec_in, dec_h)
                    predicted[t] = dec_out
                    dec_in = dec_out

            if method == 'teacher_forcing':
                # use teacher forcing
                if random.random() < tfr:
                    for t in range(target_len):
                        dec_out, dec_h = self.decoder(dec_in, dec_h)
                        predicted[t] = dec_out
                        dec_in = target[t, :].unsqueeze(1)
                # predict recursively
                else:
                    for t in range(target_len):
                        dec_out, dec_h = self.decoder(dec_in, dec_h)
                        predicted[t] = dec_out
                        dec_in = dec_out

            if method == 'mixed_teacher_forcing':
                # predict using mixed teacher forcing
                for t in range(target_len):
                    dec_out, dec_h = self.decoder(dec_in, dec_h)
                    predicted[t] = dec_out
                    # predict with teacher forcing
                    if random.random() < tfr:
                        dec_in = target[t, :].unsqueeze(1)
                    # predict recursively
                    else:
                        dec_in = dec_out

            loss = criterion(predicted.squeeze(2), target)
            loss.backward()
            optimizer.step()

            val_predicted = self.predict(val, val_target.size(0))
            val_loss = criterion(val_predicted.squeeze(2), val_target)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_params = copy.deepcopy(self.state_dict())

            losses[e] = loss.item()

            if e % 10 == 0:
                print(f'Epoch {e}/{epochs}| '
                      f'test: {round(loss.item(), 4)}, '
                      f'val: {round(val_loss.item(), 4)}')

            # dynamic teacher forcing
            if dynamic_tf and tfr > 0:
                tfr = tfr - 0.02

        return best_model_params, best_val_loss

    def predict(self, x, target_len):
        y = torch.zeros(target_len, x.shape[1], 1)

        _, enc_h = self.encoder(x)
        dec_in = x[-1, :, 0].unsqueeze(1)
        dec_h = enc_h

        for t in range(target_len):
            dec_out, dec_h = self.decoder(dec_in, dec_h)
            y[t] = dec_out
            dec_in = dec_out

        return y
