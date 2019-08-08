import argparse
import os
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
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import mlflow
import time




"""
README

Batch count indicates how many batches are run per epoch.
The small size of the dataset necessitates a batch size
larger than the size of the dataset, requiring the specification
of a batch count. Num epochs no longer indicates the amount of passes
through the set, and has been kept for convention and convienience.

"""

def train(dataloader, device, num_epochs,
             real_min,
             real_max,
             fake_min,
             fake_max,
             label_function,
             criterion,
             fixed_noise,
             netG, netD,
             lrSchedulerD, lrSchedulerG,
             optimizerD, optimizerG, nz, start_epoch, writer, batch_count, size):

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    run_avg_Gloss = 0
    run_avg_Dloss = 0
    avg_count = 1



    print("Starting Training Loop...")
    start_time = time.time()
    # For each epoch
    for epoch in np.arange(start_epoch, num_epochs+start_epoch):
        # For each batch in the dataloader
        # print(len(dataloader))
        for i, data in enumerate(dataloader):
            # print(len(dataloader))
            # print(list(enumerate(dataloader)))
            # print(i)


            ##### Label Smoothing #####
            real_label = label_function(real_min, real_max)
            # real_label = 0
            fake_label = label_function(fake_min, fake_max)
            # fake_label = 1

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            if i == 0 and epoch == 0:
                print()
                print('Input image size in pixels is {}'.format(size))
                print('Num epochs is {}'.format(num_epochs))
                print('Batch count is {}'.format(len(dataloader)))
                print("Batch size is {}".format(b_size))
                print()

            # Forward pass real batch through D
            output = netD(real_cpu)
            # print(output.shape)
            output = output.view(-1)
            # print(output.shape)

            label_size = list(output.shape)[0]

            # construct the matrix of labels
            label = torch.zeros(label_size,device=device)
            random_indicies = np.random.randint(0,high=label_size-1,size=int(np.round(label_size*0.05, 0)))
            for ix in np.arange(label.size()[0]):
                # add random noise to the discriminator (deliberately make 5% of labels incorrectly indicate fake)
                if ix in random_indicies:
                    label[ix] = label_function(fake_min, fake_max)
                # ... otherwise add a noisey real label
                else:
                    label[ix] = label_function(real_min, real_max)
            # if epoch == 1 and i == 1:
                # print(label)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(label_size, nz, 1, 1, device=device)
            # print(noise)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            # optimizerD.step()
            lrSchedulerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            # optimizerG.step()
            lrSchedulerG.step()

            # Output training stats
            # if i % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs+start_epoch, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


            # append current loss to tensorboard
            writer.add_scalar("Generator Loss", errG.item(), global_step=iters)
            writer.add_scalar("Discriminator Loss", errD.item(), global_step=iters)

            # append current average loss to tensorboard
            run_avg_Gloss = (run_avg_Gloss + errG.item() / avg_count)
            run_avg_Dloss = (run_avg_Dloss + errD.item() / avg_count)
            writer.add_scalar("Generator Avg Loss", run_avg_Gloss, global_step=iters)
            writer.add_scalar("Discriminator Avg Loss", run_avg_Dloss, global_step=iters)
            avg_count += 1

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

            if (epoch == (start_epoch+num_epochs) - 1) and (i == batch_count - 1):
                torch.save({
                            'epoch': epoch,
                            'model_D_state_dict': netD.state_dict(),
                            'lr_D_state_dict': lrSchedulerD.state_dict(),
                            'loss_D': errD
                            },
                            # f"C:/Development/footprint_generator/disc_models/gen_checkpoint_{epoch}.pth" )
                            # "/Users/alexalmond/Desktop/footprint_generator/disc_models/disc_checkpoint_{}.pth".format(epoch))
                            f"C:/Development/footprint_generator/disc_models/disc_checkpoint_size_{size}_epoch_{epoch}.pth")
                torch.save({
                            'epoch': epoch,
                            'model_G_state_dict': netG.state_dict(),
                            'lr_G_state_dict': lrSchedulerG.state_dict(),
                            'loss_G': errG,
                            },
                            # f"C:/Development/footprint_generator/gen_models/disc_checkpoint_{epoch}.pth"
                            # "/Users/alexalmond/Desktop/footprint_generator/gen_models/gen_checkpoint_{}.pth".format(epoch)
                            f"C:/Development/footprint_generator/gen_models/gen_checkpoint_size_{size}_epoch_{epoch}.pth")
                # return [dataloader, img_list, G_losses, D_losses]
            if epoch == 0 and i == 0:
                print('')
                print(f"One batch takes {time.time() - start_time} seconds to process")
                print('')



    return [dataloader, img_list, G_losses, D_losses]


def loss_curve(G_losses, D_losses):
    loss = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return loss

def image_results(dataloader, img_list, device):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
