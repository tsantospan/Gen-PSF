from __future__ import print_function
# %matplotlib inline
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import argparse

from wgan_gradient_penalty import *


def train(nz=20, training_s="1619", mode_eq="log"):
    # nz is the latent dimension
    # training_s is the training set used : "1719" for K12 / "1619" and "1617" for H23
    # mode_eq is to chose the equalization (white or log)
    
    bs = 64 # batch size

    # Root directory for dataset
    if training_s == "1619": #H23 
        dataroot = "../../data/2016_to_2019/base_train/"+mode_eq+"_train/" 
    elif training_s == "1617": #H23
        dataroot = "../../data/2016_2017/base_train/"+mode_eq+"_train/" 
    elif training_s == "1719": #K12
        dataroot = "../../data/k12_2017_to_2019/base_train/"+mode_eq+"_train/" 

    if training_s == "1719":
        suffix = "_k12"
    else:
        suffix = ""

    # Gain of the detector
    gamma = 1.75
    # Statistics on the PSF sums (alphas) and number of frames (N)
    alphas_Ns = torch.Tensor(np.load("../../data/alphas_Ns_"+training_s+suffix+".npy"))
    Ns = alphas_Ns[1,:]
    alphas = alphas_Ns[0,:]

    # Mean and std images (for whitening)
    MEAN = torch.Tensor(np.load("../../data/MEAN_"+training_s+suffix+".npy")).to(device)
    STD = torch.Tensor(np.load("../../data/STD_"+training_s+suffix+".npy")).to(device)

    # Read-out noise maps
    ron0 = torch.Tensor(np.load("../../data/ron1_0_128.npy").astype(np.float32))
    ron1 = torch.Tensor(np.load("../../data/ron1_1_128.npy").astype(np.float32))




    # Define whitening and unwhitening
    bound_mini=-3.5
    bound_maxi=9
    def whitening(PSF): # Simple whitening
        return (PSF-MEAN)/STD

    def whitening_complete(X): # With scaling to [0,1]
        return (whitening(X)-bound_mini)/(bound_maxi-bound_mini)

    def unwhite(X): # Unwhithening
        return (X*(bound_maxi-bound_mini)+bound_mini)*STD+MEAN
        

    # Log equalization for visualization
    def logeq(Im):
        return torch.log(1000*Im+1)/torch.log(1001)

    def normminmax(Im):
        return (Im - torch.min(Im))/(torch.max(Im)-torch.min(Im))

    def viseq(Im):
        return logeq(normminmax(Im))

    # Log-equalization for the NN (with torch)
    log_10001 = torch.log(torch.Tensor([10001])).to(device)
    def logeq2(Im):
        return torch.log(10000*Im+1)/log_10001+0.3

    def unlognorm2(Im):
        return (10001**(Im-0.3)-1)/10000

    clamp_log_low = unlognorm2(0) # Limits to put all between 0 and 1 after unequalization
    clamp_log_high = unlognorm2(1)


    # Generate the noise on a whole batch
    def generate_noise_batch(PSFs, batch_size=64):
        index = torch.randint(0,alphas_Ns.shape[1], (batch_size,))
        
        choice = np.random.rand()
        if choice<0.5:
            the_ron = ron0.to(device)
        else:
            the_ron = ron1.to(device)

        N_batch = torch.reshape(Ns[index], (batch_size, 1, 1, 1)).to(device) # Pick number of frames
        alpha_batch = torch.reshape(alphas[index], (batch_size, 1, 1, 1)).to(device) # Pick PSF sum

        variance = ((the_ron/alpha_batch)**2/N_batch+gamma/N_batch/alpha_batch*abs(PSFs.detach())) # Good version
        
        std = torch.sqrt(variance)
        
        try: # In case std is nan
            noise = torch.normal(0, std)
        except: # For debugging
            for i in range(64):
                try:
                    A = torch.normal(0, std[i,0])
                except:
                    plt.show()
                    print("pb with", i)
                    plt.imshow(PSFs[i,0].detach().cpu())
                    plt.colorbar()
                    plt.show()
        
        return noise
        
        
    # Noiser for whitening
    def add_noise_batch_whitened(PSFs):
        og = unwhite(PSFs)
        noise = generate_noise_batch(og, batch_size)
        new = og + noise
        new = whitening_complete(new)
        new = torch.clamp(new, 0, 1)
        return new
    # Noiser for log-equalization
    def add_noise_batch_logged(PSFs, batch_size=64):
        og = unlognorm2(PSFs)
        noise = generate_noise_batch(og, batch_size)
        new = og + noise
        new = torch.clamp(new, clamp_log_low, clamp_log_high)
        new = logeq2(new)
        return new
        
        

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 64

    # Spatial size of training images. All images will be resized to this
    # size using a transformer if needed.
    image_size = 128

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1



    # Check the choice of the equalization
    white_gen = (mode_eq == "white")
    if white_gen:
        uneq = unwhite
        eq = whitening_complete
        noiser = add_noise_batch_whitened
    else:
        uneq = unlognorm2
        eq = logeq2
        noiser = add_noise_batch_logged
        
    suite_name_model  = "_train"+training_s+"_nlat"+str(nz)+"_"+mode_eq
    if training_s == "1719":
        suite_name_model = "_K12" + suite_name_model
    else:
        suite_name_model = "_H23" + suite_name_model

    wgan = WGAN_GP(nb_channels=1, cud=True, nb_gen_iter=50000, name_suite=suite_name_model, n_latent=nz, noiser=noiser, eq=eq, uneq=uneq, white=white_gen)

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Grayscale()
                            ])) # Some transforms can be useless is we use the code adequatly; there are here for safety.


    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
                                                                                    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    print("Device : ", device)
    wgan.train(dataloader)

    wgan.plot_losses()


def get_args():
    parser = argparse.ArgumentParser(description='Train the network, given latent dimensions, training set, and equalization')

    parser.add_argument('--nz', type=int, default=20, help='Number of latent dimensions')
    parser.add_argument('--training_s', type=str, default="1619", help='Training set (1617/1619/1719)')
    parser.add_argument('--mode_eq', type=str, default="log", help='Equalization (white/log)')


if __name__ == '__main__':
    args = get_args()

    train(args.nz, args.training_s,args.mode_eq)