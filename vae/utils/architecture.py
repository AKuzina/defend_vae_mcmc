import torch
import torch.nn as nn
from vae.utils.nn import ConvBlock


class EncoderMnist(nn.Module):
    def __init__(self, z_dim, in_ch, **kwargs):
        super(EncoderMnist, self).__init__()

        wn = False
        act = nn.ReLU()
        ch = [in_ch, in_ch, in_ch*2, in_ch*4, z_dim//64]

        self.q_z_layers = nn.Sequential(
            ConvBlock(1, ch[0], 3, stride=1, padding=0, act=act, weight_norm=wn),  # 28->26
            ConvBlock(ch[0], ch[1], 3, stride=2, padding=0, act=act, weight_norm=wn),  #26->12
            ConvBlock(ch[1], ch[2], 5, stride=1, padding=0, act=act, weight_norm=wn),  #12->8
            ConvBlock(ch[2], ch[3], 3, stride=1, padding=1, act=act, weight_norm=wn),  #8->8
        )

        self.q_z_mean = nn.Sequential(
            ConvBlock(ch[3], ch[4], 5, stride=1, padding=2, act=None, weight_norm=wn),  #8->8
            nn.Flatten()
        )
        self.q_z_logvar = nn.Sequential(
            ConvBlock(ch[3], ch[4], 5, stride=1, padding=2, act=None, weight_norm=wn),  #8->8
            nn.Flatten()
        )
        self.init()

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.reshape(-1, 1, 28, 28)
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)


class DecoderMnist(nn.Module):
    def __init__(self, z_dim, in_ch, **kwargs):
        super(DecoderMnist, self).__init__()

        self.chnl = [z_dim//64, in_ch*4, in_ch*2, in_ch]
        self.p_x_layers = nn.Sequential(
            nn.ConvTranspose2d(self.chnl[0], self.chnl[1], 3, stride=1, padding=1),  # 8->8
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[1], self.chnl[2], 4, stride=1, padding=0),  # 8->11
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[2], self.chnl[3], 4, stride=1, padding=0),  # 8->14
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[3], 1, 4, stride=2, padding=1),  # 14->28
            nn.Sigmoid()
        )
        self.init()

    def forward(self, z):
        if len(z.shape) < 3:
            z = z.reshape(-1, self.chnl[0], 8, 8)
        x_mean = self.p_x_layers(z)
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)


def get_architecture(args):
    params_enc = {'z_dim': args.z_dim, 'in_ch': args.num_ch}
    params_dec = {'z_dim': args.z_dim, 'in_ch': args.num_ch}

    if 'mnist' in args.dataset_name:
        if args.model == 'conv':
            enc = EncoderMnist
            dec = DecoderMnist
        else:
            raise NotImplementedError(
                "The model {} is not implemented".format(config.model)
            )
    else:
        raise NotImplementedError(
            "The dataset {} does not exist".format(config.dataset_name)
        )
    arc_getter = lambda: (enc(**params_enc), dec(**params_dec))
    return arc_getter, args