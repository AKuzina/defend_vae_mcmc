import torch
import torch.nn as nn
from vae.utils.nn import ConvBlock


class EncoderMnist(nn.Module):
    def __init__(self, z_dim, in_ch, long=False, **kwargs):
        super(EncoderMnist, self).__init__()

        wn = False
        act = nn.ReLU()
        ch = [in_ch, in_ch*2, in_ch*3, in_ch*4, z_dim//4]
        conv_arg = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        if long:
            ch = [in_ch, in_ch*2, in_ch*3, in_ch*4, z_dim]
            conv_arg = {'kernel_size': 3, 'stride': 2, 'padding': 1}

        self.q_z_layers = nn.Sequential(
            ConvBlock(1, ch[0], 3, stride=2, padding=1, act=act, weight_norm=wn),  # 28->14
            ConvBlock(ch[0], ch[1], 3, stride=2, padding=1, act=act, weight_norm=wn),  #14->7
            ConvBlock(ch[1], ch[2], 3, stride=2, padding=1, act=act, weight_norm=wn),  #7->4
            ConvBlock(ch[2], ch[3], 3, stride=2, padding=1, act=act, weight_norm=wn),  #4->2
        )

        self.q_z_mean = nn.Sequential(
            ConvBlock(ch[3], ch[4], act=None, weight_norm=wn,  **conv_arg),  #2->2/1
            nn.Flatten()
        )
        self.q_z_logvar = nn.Sequential(
            ConvBlock(ch[3], ch[4], act=None, weight_norm=wn,  **conv_arg),  #2->2/1
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
    def __init__(self, z_dim, in_ch, long=False, **kwargs):
        super(DecoderMnist, self).__init__()

        self.chnl = [z_dim//4, in_ch*4, in_ch*3, in_ch*2, in_ch]
        conv_arg = {'kernel_size': 3, 'stride': 2, 'padding': 0}
        self.z_shape = 2
        if long:
            self.chnl = [z_dim, in_ch*4, in_ch*3, in_ch*2, in_ch]
            conv_arg = {'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation':2}
            self.z_shape = 1

        self.p_x_layers = nn.Sequential(
            nn.ConvTranspose2d(self.chnl[0], self.chnl[1], **conv_arg),  # 2/1 -> 5
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[1], self.chnl[2], 3, stride=1, padding=0),  # 5->7
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[2], self.chnl[3], 3, stride=1, padding=1),  # 7->7
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[3], self.chnl[4], 4, stride=2, padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[4], 1, 4, stride=2, padding=1),  # 14->28
            nn.Sigmoid()
        )
        self.init()

    def forward(self, z):
        if len(z.shape) < 3:
            z = z.reshape(-1, self.chnl[0], self.z_shape, self.z_shape)
        x_mean = self.p_x_layers(z)
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)


class EncoderCelebA(EncoderMnist):
    def __init__(self, z_dim, in_ch, **kwargs):
        super(EncoderCelebA, self).__init__(z_dim, in_ch)
        act = nn.ReLU()
        self.z_dim = z_dim
        self.blocks = []
        self.chnl = [3, in_ch, in_ch*2, in_ch*4, in_ch*8, z_dim]

        self.q_z_layers = nn.Sequential(
            ConvBlock(self.chnl[0], self.chnl[1], 3, stride=2, padding=1,
                      batchnorm=True, act=act),  # 32 -> 16
            ConvBlock(self.chnl[1], self.chnl[2], 3, stride=2, padding=1,
                      batchnorm=True, act=act),  # 16 -> 8
            ConvBlock(self.chnl[2], self.chnl[3], 3, stride=2, padding=1,
                      batchnorm=True, act=act),  # 8 -> 4
            ConvBlock(self.chnl[3], self.chnl[4], 3, stride=2, padding=1,
                      batchnorm=True, act=act),  # 4 -> 2
        )

        self.q_z_mean = nn.Sequential(
            ConvBlock(self.chnl[4], self.chnl[5], 3, stride=2, padding=1, act=None),  # 2 -> 1
            nn.Flatten()
        )
        self.q_z_logvar = nn.Sequential(
            ConvBlock(self.chnl[4], self.chnl[5], 3, stride=2, padding=1, act=None),  # 2 -> 1
            # nn.Hardtanh(-4., 4.),
            nn.Flatten()
        )
        self.init()

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.reshape(-1, 3, 32, 32)
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar


class DecoderCelebA(DecoderMnist):
    def __init__(self, z_dim, in_ch, **kwargs):
        super(DecoderCelebA, self).__init__(z_dim, in_ch)
        self.z_dim = z_dim
        self.z_shape = 1
        self.chnl = [z_dim, in_ch*8, in_ch*4, in_ch*2, in_ch, 3]

        self.p_x_layers = nn.Sequential(
            nn.ConvTranspose2d(self.chnl[0], self.chnl[1], 3, stride=1, padding=0, dilation=2),  # 1 -> 5
            nn.BatchNorm2d(self.chnl[1], affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[1], self.chnl[2], 3, stride=2, padding=2, output_padding=1),  # 5->8
            nn.BatchNorm2d(self.chnl[2], affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[2], self.chnl[3], 3, stride=2, padding=1, output_padding=1),  # 8->16
            nn.BatchNorm2d(self.chnl[3], affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.chnl[3], self.chnl[4], 3, stride=2, padding=1, output_padding=1),  # 16->32
            nn.BatchNorm2d(self.chnl[4], affine=True),
            nn.ReLU()
        )
        self.x_mu = nn.Sequential(
            nn.Conv2d(self.chnl[4], self.chnl[5], kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.x_logvar = nn.Sequential(
            nn.Conv2d(self.chnl[4], self.chnl[5], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Hardtanh(-8., 8.)
        )
        self.init()

    def forward(self, z):
        if len(z.shape) < 3:
            z = z.reshape(-1, self.chnl[0], self.z_shape, self.z_shape)
        h = self.p_x_layers(z)
        x_mean = self.x_mu(h)
        x_logvar = self.x_logvar(h)
        return x_mean, x_logvar


def get_architecture(args):
    params_enc = {'z_dim': args.z_dim, 'in_ch': args.num_ch, 'long': args.latent_long}
    params_dec = {'z_dim': args.z_dim, 'in_ch': args.num_ch, 'long': args.latent_long}

    if 'mnist' in args.dataset_name:
        if args.model in ['conv', 'vcd']:
            enc = EncoderMnist
            dec = DecoderMnist
        else:
            raise NotImplementedError(
                "The model {} is not implemented".format(config.model)
            )
    elif 'celeba' in args.dataset_name:
        if args.model in ['conv', 'vcd']:
            enc = EncoderCelebA
            dec = DecoderCelebA
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