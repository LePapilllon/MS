#!/usr/bin/python3
import os
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

dataroot = 'datasets/lh/'
cuda = True
input_nc = 3
output_nc = 3
size = 512
n_cpu =4
generator_A2B = 'output/netG_A2B.pth'

###### Definition of variables ######
# Networks
netG_A2B = Generator(input_nc, output_nc)

if cuda:
    netG_A2B.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(generator_A2B))
# netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
# Set model input
real_A = Image.open(os.path.join(dataroot, 'test/A', 'real_image.tif'))
img_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
real_A = img_transforms(real_A).cuda()
real_A = real_A.unsqueeze(0)
# Generate output
print(real_A.shape)
fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
print(fake_B.shape)
fake_B = fake_B.squeeze(0)
print(fake_B.shape)
# Save image files
save_image(fake_B, 'output/lh/fake_image.tif' )
###################################
