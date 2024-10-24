import torch
import torch.nn as nn

class FLD(nn.Module):
    def __init__(self,
                 observation_dim,
                 observation_horizon,
                 latent_channel,
                 device,
                 dt=0.02,
                 encoder_shape=None,
                 decoder_shape=None,
                 **kwargs,
                 ):
        if kwargs:
            print("FLD.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(FLD, self).__init__()
        self.input_channel = observation_dim
        self.observation_horizon = observation_horizon
        self.latent_channel = latent_channel
        self.device = device
        self.dt = dt

        self.args = torch.linspace(-(observation_horizon - 1) * self.dt / 2, (observation_horizon - 1) * self.dt / 2, self.observation_horizon, dtype=torch.float, device=self.device)
        self.freqs = torch.fft.rfftfreq(observation_horizon, device=self.device)[1:] * observation_horizon
        self.encoder_shape = encoder_shape if encoder_shape is not None else [int(self.input_channel / 3)]
        self.decoder_shape = decoder_shape if decoder_shape is not None else [int(self.input_channel / 3)]        
        
        encoder_layers = []
        curr_in_channel = self.input_channel
        for hidden_channel in self.encoder_shape:
            encoder_layers.append(
                nn.Conv1d(
                    curr_in_channel, 
                    hidden_channel, 
                    observation_horizon, 
                    stride=1, 
                    padding=int((observation_horizon - 1) / 2), 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros')
                )
            encoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
            encoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel
        encoder_layers.append(
            nn.Conv1d(
                self.encoder_shape[-1], 
                latent_channel, 
                observation_horizon, 
                stride=1, 
                padding=int((observation_horizon - 1) / 2), 
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode='zeros')
            )
        encoder_layers.append(nn.BatchNorm1d(num_features=latent_channel))
        encoder_layers.append(nn.ELU())

        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        self.encoder.train()

        self.phase_encoder = nn.ModuleList()
        for _ in range(latent_channel):
            phase_encoder_layers = []
            phase_encoder_layers.append(nn.Linear(observation_horizon, 2))
            phase_encoder_layers.append(nn.BatchNorm1d(num_features=2))
            phase_encoder = nn.Sequential(*phase_encoder_layers).to(self.device)
            self.phase_encoder.append(phase_encoder)
        self.phase_encoder.train()

        decoder_layers = []
        curr_in_channel = latent_channel
        for hidden_channel in self.decoder_shape:
            decoder_layers.append(
                nn.Conv1d(
                    curr_in_channel, 
                    hidden_channel, 
                    observation_horizon, 
                    stride=1, 
                    padding=int((observation_horizon - 1) / 2), 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros')
                )
            decoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
            decoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel
        decoder_layers.append(
            nn.Conv1d(
                self.decoder_shape[-1], 
                self.input_channel, 
                observation_horizon, 
                stride=1, 
                padding=int((observation_horizon - 1) / 2), 
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode='zeros')
            )
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        self.decoder.train()

    def forward(self, x, k=1):
        if k == 0:
            k = 1
        x = self.encoder(x)
        latent = x
        frequency, amplitude, offset = self.fft(x)
        phase = torch.zeros((x.size(0), self.latent_channel), device=self.device, dtype=torch.float)
        for i in range(self.latent_channel):
            phase_shift = self.phase_encoder[i](x[:, i, :])
            phase[:, i] = torch.atan2(phase_shift[:, 1], phase_shift[:, 0]) / (2 * torch.pi)
        
        params = [phase, frequency, amplitude, offset] # (batch_size, latent_channel)

        phase_dynamics = phase.unsqueeze(0) + frequency.unsqueeze(0) * self.dt * torch.arange(0, k, device=self.device, dtype=torch.float, requires_grad=False).view(-1, 1, 1) # (k, batch_size, latent_channel)
        z = amplitude.unsqueeze(-1).unsqueeze(0) * torch.sin(2 * torch.pi * ((frequency.unsqueeze(-1) * self.args).unsqueeze(0) + phase_dynamics.unsqueeze(-1))) + offset.unsqueeze(-1).unsqueeze(0) # (k, batch_size, latent_channel, observation_horizon)
        signal = z[0]
        pred_dynamics = self.decoder(z.flatten(0, 1)).view(k, -1, self.input_channel, self.observation_horizon) # (k, batch_size, input_channel, observation_horizon)

        return pred_dynamics, latent, signal, params

    def fft(self, x):
        rfft = torch.fft.rfft(x, dim=2)
        magnitude = rfft.abs()
        spectrum = magnitude[:, :, 1:]
        power = torch.square(spectrum)
        frequency = torch.sum(self.freqs * power, dim=2) / torch.sum(power, dim=2)
        amplitude = 2 * torch.sqrt(torch.sum(power, dim=2)) / self.observation_horizon
        offset = rfft.real[:, :, 0] / self.observation_horizon
        return frequency, amplitude, offset

    def get_dynamics_error(self, state_transitions, k):
        self.eval()
        state_transitions_sequence = torch.zeros(
            state_transitions.size(0),
            state_transitions.size(1) - self.observation_horizon + 1,
            self.observation_horizon,
            state_transitions.size(2),
            dtype=torch.float,
            device=self.device,
            requires_grad=False
            )
        for step in range(state_transitions.size(1) - self.observation_horizon + 1):
            state_transitions_sequence[:, step] = state_transitions[:, step:step + self.observation_horizon, :]
        with torch.no_grad():
            pred_dynamics, _, _, _ = self.forward(state_transitions_sequence.flatten(0, 1).swapaxes(1, 2), k)
        pred_dynamics = pred_dynamics.swapaxes(2, 3).view(k, -1, state_transitions.size(1) - self.observation_horizon + 1, self.observation_horizon, state_transitions.size(2))
        error = torch.zeros(state_transitions.size(0), device=self.device, dtype=torch.float, requires_grad=False)
        for i in range(k):
            error[:] += torch.square((pred_dynamics[i, :, :state_transitions_sequence.size(1) - i] - state_transitions_sequence[:, i:])).mean(dim=(1, 2, 3))
        return error
