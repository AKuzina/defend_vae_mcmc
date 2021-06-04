import torchvision.models as models
import torch.nn as nn
import torch


# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(nn.Module):
    def __init__(self, target_img, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        blocks.append(models.vgg16(pretrained=True).features[23:30].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resize = resize
        self.target_img = target_img
        self.target = None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def get_statistics(self, img):
        # nomalize to [-1, 1]
        img = (img-self.mean) / self.std

        stats = []
        if self.resize:
            img = self.transform(img)
        for block in self.blocks:
            img = block(img)
            stats.append(img)
        return stats

    def forward(self, noisy):
        loss = 0.0
        if self.target is None:
            self.target = self.get_statistics(self.target_img)
        curr_status = self.get_statistics(noisy)
        for x, y in zip(curr_status, self.target):
            loss += torch.abs(x - y).mean([1,2,3])
        return loss