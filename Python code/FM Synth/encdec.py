import torch
import torch.nn.functional as F
import torch.nn as nn


# Encoder class
class Encoder(nn.Module):
    def __init__(self, cardinality_list, input_shape='4sec'):
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.cardinality_list = cardinality_list
        self.ch = 1

        self.encoder_out_dim = sum(cardinality_list.values())

        self.enc_nn = nn.Sequential(
            nn.Conv2d(self.ch, 8, (5, 5), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(8, 16, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(16, 32, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(32, 64, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(64, 128, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(128, 256, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.features_mixer_cnn = nn.Sequential(
            nn.Conv2d(256, 512, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(512, 2048, (1, 1), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=12288, out_features=self.encoder_out_dim),
            nn.BatchNorm1d(self.encoder_out_dim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        x = self.enc_nn(x)
        x = self.features_mixer_cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x


# Decoder class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=104, out_features=24576, bias=True),
            nn.Dropout(p=0.3, inplace=False)
        )

        self.features_unmixer_cnn = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.single_ch_cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(8, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(8, 1, kernel_size=(5, 5), stride=(2, 1), padding=(2, 3)),
            nn.Hardtanh(min_val=-1.0, max_val=1.0)
        )

    def forward(self, v_in):

        x = self.mlp(v_in)
        x = x.view(-1, 2048, 3, 4)
        x = self.features_unmixer_cnn(x)
        x = self.single_ch_cnn(x)
        return x



param_buckets_tal = {
        'osc1_wave': 3, 'osc1_freq': 12, 'osc1_mod_index': 20, 'lfo1_freq': 31, 'lfo1_wave': 4,
        'am_mod_wave': 5, 'am_mod_freq': 8, 'am_mod_amount': 5, 'filter_freq': 16
    }

# EncDec class
class EncDec(nn.Module):
    def __init__(self):
        super(EncDec, self).__init__()
        self.encoder = Encoder(param_buckets_tal)
        self.decoder = Decoder()

    def get_estimated_params(self, spec_in, is_dict=True):
        v_out = self.encoder(spec_in)
        if is_dict:
            dict_out = {}
            from_idx = 0
            for f in param_buckets_tal:
                to_index = from_idx + param_buckets_tal[f]
                dict_out[f] = v_out[:, from_idx:to_index]
                from_idx = to_index
            return dict_out
        return v_out

    def get_estimated_spec(self, v_in):
        " v_in is [Nx104] torch "
        return self.decoder(v_in)

    def forward(self, spec_in):
        v_out = self.encoder(spec_in)
        spec_out = self.decoder(v_out)
        return spec_out


