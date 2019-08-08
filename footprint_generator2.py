import argparse
import torch
import torchvision
from torch import nn, optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML
import torchvision.datasets as dset
from torchvision.datasets import ImageFolder
import training_script2 as ts
import subprocess
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import rotation
import math
import cv2
import mlflow


"""

README

Techniques Used:

Label Smoothing: Real labels between 0.9 and 1.0, fake between 0.0 and 0.1
Random Noise for Discrimnator: 5% of discriminator labels are incorrect
Learning Rate Scheduler: reduced learning rate after specefied epoch count
Model Architecture: taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Transformations:
    Crops (~90% of original image)
    Vertical and Horizontal Flips
    Rotations in degree range [0, 360]

Install pytroch 1.14 nightly to use TensorBoard live monitering
"""

##########################################################################################################################
##############################################  Custom Transformations  ##################################################
##########################################################################################################################

class PercentageCrop(object):
    """ Takes a percentage crop of an image """
    def __init__(self, percentage):
        assert percentage >= 0.75, "The image crop is too small to capture patterns to be learned"
        self.percentage = percentage
        self.reduction_factor = np.sqrt(percentage)

    def __call__(self, image):
        w, h = image.size
        new_w = np.round(self.reduction_factor * w, 0)
        new_h = np.round(self.reduction_factor * h, 0)
        w_diff = w - new_w
        h_diff = h - new_h

        location = random.choice(['center', 'upper_right', 'lower_right', 'lower_left', 'upper_left'])

        # (left, upper, right, lower)
        if location == 'center':
            cropped_image = image.crop((w_diff, h_diff, w - w_diff, h - h_diff))
        elif location == 'upper_right':
            cropped_image = image.crop((2*w_diff, 0, w, h - (2*h_diff)))
        elif location == 'lower_right':
            cropped_image = image.crop((2*w_diff, 2*h_diff, w, h))
        elif location == 'lower_left':
            cropped_image = image.crop((0, 2*h_diff, w - (2*w_diff), h))
        else:
            cropped_image = image.crop((0, 0, w - (2*w_diff), h - (2*h_diff)))

        return cropped_image

class RotateAndResize(object):
    """ Rotates an image and removes black corners that appear as a result of the rotation """

    def __call__(self, image):

        # convert PIL image to openCV image
        cv2_image = PIL_to_cv2(image)

        image_height, image_width = cv2_image.shape[0:2]

        i = np.random.randint(1, high=360)
        image_rotated = rotation.rotate_image(cv2_image, i)
        image_rotated_cropped = rotation.crop_around_center(
            image_rotated,
            *rotation.largest_rotated_rect(
                image_width,
                image_height,
                math.radians(i)
            )
        )

        # convert openCV image back to PIL
        image_rotated_cropped_PIL = cv_to_PIL(image_rotated_cropped)

        return image_rotated_cropped_PIL


"""
Helper function to convert PIL images to cv2 images
"""
# adapted from https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def PIL_to_cv2(PIL_image):

    pil_image = PIL_image.convert('RGB')
    open_cv_image = np.array(pil_image)

    # Convert RGB to BGR
    cv2_image = open_cv_image[:, :, ::-1].copy()

    return cv2_image

"""
Helper function to convert cv2 images to PIL images
"""
# adapted from https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format?noredirect=1&lq=1
def cv_to_PIL(cv2_image):

    #Convert BGR to RGB
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil


##############################################################################################################################################
####################################################### Neural Net Code ######################################################################
##############################################################################################################################################

""" Weight Initilization """

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

"""
The generator side of the net. Attempts to generate images that are close
to what it has learned.

"""
class suburbGenerator(nn.Module):
    def __init__(self, ngpu):
        super(suburbGenerator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

"""
The judgement side of the net. Tries to discriminate between real images and fakes,
attempting to correctly identify them.
"""

class suburbDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(suburbDiscriminator, self).__init__()
        self.ngpu = ngpu

        # second input to Conv2d is the number of filters
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        # print(input.shape)
        return self.main(input)

#############################################################################################################################################################
################################################################## initialize network parameters ############################################################
#############################################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('size', help='size of the input image in pixes (Valid sizes are 8,16,32,64,128)')
parser.add_argument('epochs', help='number of epochs for model (convention)')
parser.add_argument('batch_count', help='number of batches per epoch')
parser.add_argument('batch_size', help='batch size for training')
parser.add_argument('learning_rate', help='learning rate of model')
parser.add_argument('--G_path', default=None, help='path to a pretrained generator model')
parser.add_argument('--D_path', default=None, help='path to a pretrained discriminator model')
parser.add_argument('--G_lr', default=None, help='path to lr scheduler for generator')
parser.add_argument('--D_lr', default=None, help='path to lr scheduler for the discriminator')

# dataroot = '/Users/alexalmond/Desktop/footprint_generator/'
dataroot = 'C:/Development/footprint_generator'
start_epoch = 0 # start epoch for training defaults to 0

args = parser.parse_args()

global size # size of the input images in pixels
size = int(args.size)

global nc  # number of colour channels for the images
nc = 3

global nz  # size of generator input (vector)
nz = 100

global ngf # size of feature maps in the generator
ngf = size

global ndf # size of feature maps in the discriminator
ndf = size

global beta1 # Beta1 Hyperparam for Adam optimizers
beta1 = 0.5

global ngpu  # number of GPUs available
ngpu = 0

# number of epochs
global num_epochs
num_epochs = int(args.epochs)
# number of batches per epoch
global batch_count
batch_count = int(args.batch_count)
# batch size
global batch_size
batch_size = int(args.batch_size)
# learning rate for the optimizer
# 0.0002 is reccommended
global lr
lr = float(args.learning_rate)
#############################################################################################################################################################################
############################################################### initialize the dataset, dataloader, and networks ############################################################
#############################################################################################################################################################################

# device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
device = torch.device('cpu')
print('Device is {}'.format(device))

dataset = ImageFolder(root=dataroot,
                           transform = transforms.Compose([

                           # take a crop of the image (92% of original size)
                           PercentageCrop(0.92),

                           # perfrom a rotation and crop to remove blackspace
                           RotateAndResize(),

                           # filp the image horizonally
                           transforms.RandomHorizontalFlip(p=0.5),

                           # filp the image vertically
                           transforms.RandomVerticalFlip(p=0.5),

                           # resize the image to make it square
                           transforms.Resize((size, size)),

                           # convert the image to a tensor
                           transforms.ToTensor(),

                           # normalize the image
                           transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                                        ]))

sampler = [np.random.choice(np.arange(0,286), batch_size) for i in np.arange(batch_count)]

dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)

netG = suburbGenerator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.dataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

print(netG)

netD = suburbDiscriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.dataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

print(netD)

#######################################################################################################################################################################################
############################################################ initialize the loss function and optimizer ###############################################################################
#######################################################################################################################################################################################

criterion = nn.BCELoss()

# creates batch of latent vectors to visualize the generator
fixed_noise = torch.randn(size, nz, 1, 1, device = device)

# real labels are between 0 and 0.1, fake are between 0.9 and 1.0 (label smoothing)
noisey_label = True
fake_min, fake_max = 0.0, 0.2
real_min, real_max = 0.8, 1.0
label_function = lambda a, b: (b - a) * np.random.random_sample() + a

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


# decay the learning rate by a factor of gamma every step_size number of epochs, until last_epoch number of epochs are reached
lrSchedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size = 2, gamma = 0.75, last_epoch = -1)
lrSchedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size = 2, gamma = 0.75, last_epoch = -1)


if (args.D_path and args.G_path):
    pathG = torch.load(args.G_path)
    pathD = torch.load(args.D_path)
    netG.load_state_dict(pathG['model_G_state_dict'])
    netD.load_state_dict(pathD['model_D_state_dict'])
    assert (pathG['epoch'] == pathD['epoch']), "NetG and NetD are not a pair"
    start_epoch = pathG['epoch']

if args.D_lr and args.G_lr:
    lrSchedulerD.load_state_dict(args.D_lr)
    lrSchedulerG.load_state_dict(args.G_lr)

#################################################################################################################################
############################################  Run the training script  ###########################################################
############################################  Add to tensboard, mlflow ###########################################################
#################################################################################################################################

# with mlflow.start_run():
#
#
#     # current_transforms = ['PercentageCrop(92)', 'RotateAndResize', 'RandomHorizontalFlip'
#                           # 'RandomVerticalFlip', 'Resize(64, 64)', 'ToTensor', 'Normalize']
#
#     mlflow.log_param('learning_rate', lr)
#     mlflow.log_param("num_epochs", num_epochs)
#     mlflow.log_param("batch_size", batch_size)
#     mlflow.log_param('large', large)


writer = SummaryWriter()

subprocess.Popen(['tensorboard', '--logdir=./runs', '--port', '8080'])

results = ts.train(dataloader, device, num_epochs,
             real_min,
             real_max,
             fake_min,
             fake_max,
             label_function,
             criterion,
             fixed_noise,
             netG, netD,
             optimizerD, optimizerG,
             lrSchedulerD, lrSchedulerG, nz, start_epoch, writer, batch_count, size)

# current_transforms = ['PercentageCrop(92)', 'RotateAndResize', 'RandomHorizontalFlip'
                      # 'RandomVerticalFlip', 'Resize(64, 64)', 'ToTensor', 'Normalize']

loss_curve = ts.loss_curve(results[2], results[3])
fake_batch = results[1]
real_batch = next(iter(results[0]))[0]
real_grid = torchvision.utils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu()
fake_grid = torchvision.utils.make_grid(fake_batch[-1])
writer.add_figure("Loss Curve", loss_curve)
writer.add_image("Generated Images", fake_grid)
writer.add_image('Real Images', real_grid)
# writer.add_graph(netG)
# writer.add_graph(netD)
writer.close()

ts.image_results(dataloader, results[1], device)
