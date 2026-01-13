import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import os
from torchvision import utils
import numpy as np


SAVE_PER_TIMES = 100
nchan_size = 128 # Number minimal of channels in the convolutions


# Defining equalizations for visualization

def logeq(Im):
    return torch.log(1000*Im+1)/np.log(1001)

def normminmax(Im):
    return (Im - torch.min(Im))/(torch.max(Im)-torch.min(Im))

def viseq(Im): # For visualization only
    return logeq(normminmax(Im))


class Generator(torch.nn.Module):
    def __init__(self, channels, n_latent=10, eq=None, uneq=None, white=False):
    	# We have to provide the equalization (named eq) and the unequalization (named uneq) operators
        super().__init__()
        self.n_latent = n_latent
        self.eq = eq
        self.uneq = uneq
        self.white = white # Boolean, true if equalization == whitening
        self.main_module = nn.Sequential(
 # For one more a layer (needs some other modifications)
            #nn.ConvTranspose2d(in_channels=self.n_latent,  out_channels=nchan_size*8, kernel_size=1, stride=1, padding=0, bias = True),
            #nn.ReLU(True),
            #nn.ConvTranspose2d(in_channels=nchan_size*8,  out_channels=nchan_size*16, kernel_size=4, stride=1, padding=0, bias = True),
         
            nn.ConvTranspose2d(in_channels=self.n_latent,  out_channels=nchan_size*16, kernel_size=4, stride=1, padding=0, bias = True),
            nn.BatchNorm2d(num_features=nchan_size*16),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nchan_size*16,  out_channels=nchan_size*8, kernel_size=4, stride=2, padding=1, bias = True),
            nn.BatchNorm2d(num_features=nchan_size*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nchan_size*8, out_channels=nchan_size*4, kernel_size=4, stride=2, padding=1, bias = True),
            nn.BatchNorm2d(num_features=nchan_size*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nchan_size*4, out_channels=nchan_size*2, kernel_size=4, stride=2, padding=1, bias = True),
            nn.BatchNorm2d(num_features=nchan_size*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nchan_size*2, out_channels=nchan_size, kernel_size=4, stride=2, padding=1, bias = True),
            nn.BatchNorm2d(num_features=nchan_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nchan_size, out_channels=channels, kernel_size=4, stride=2, padding=1)
            )
            # output : image (1x128x128)

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.main_module(x)
        
        if self.white:
            x = self.output(x)
        else:
            x = self.output(x)*0.7+0.3 # If we work with log equalization : we work between 0.3 and 1.0 (to avoid NaN because of simulated negative values when getting equalized again)
        
        if self.eq is not None and self.uneq is not None:
            #Unequalization
            x = self.uneq(x)

            #Normalization
            sum_ = torch.sum(x, dim=(2,3), keepdim=True) 
            x = torch.div(x,sum_)

            #Re-equalization
            x = self.eq(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.main_module = nn.Sequential(

            nn.Conv2d(in_channels=channels, out_channels=nchan_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nchan_size, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nchan_size, out_channels=nchan_size*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nchan_size*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nchan_size*2, out_channels=nchan_size*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nchan_size*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nchan_size*4, out_channels=nchan_size*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nchan_size*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nchan_size*8, out_channels=nchan_size*16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nchan_size*16, affine=True),
            nn.LeakyReLU(0.2, inplace=True) )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=nchan_size*16, out_channels=1, kernel_size=4, stride=1, padding=0),
            )


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Flatten to 16384 vector
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_GP(object):
    def __init__(self, nb_channels=1, cud=False, nb_gen_iter=40000, name_suite="", n_latent=30, noiser=None, eq=None, uneq=None, white=False):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(nb_channels, n_latent, eq, uneq, white)
        self.D = Discriminator(nb_channels)
        self.C = nb_channels
        self.n_latent = n_latent

        # To plott loss
        self.L_loss_D = [] # Loss of D
        self.L_loss_D_2 = [] # Divided by 2 (because it has to parts)
        self.L_loss_G = [] # Loss of G
        self.L_loss_D_fake = [] # Loss of D on the generated images
        self.L_loss_D_real = [] # Loss on D on the training images
        self.L_loss_D_GP = [] # Gradient Penalty loss

        self.noiser = noiser # Operator of noise simulation 
        self.uneq=uneq  # Operatior of unequalization

        self.name_suite = name_suite # For naming purpose (after the default string)

        # Check if cuda is available
        self.check_cuda(cud)


        # Hyperparameters values
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        # ADAM init
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # For evaluation
        self.number_of_images = 10
        
        # Number of training iterations (for G) 
        self.generator_iters = nb_gen_iter
        self.critic_iter = 5 # Number of steps of the critic for each step of the generator


        self.lambda_term = 10 # Scalar of gradient penalty

    
    
    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    # 
    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True, 
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False
    

    def train(self, train_loader):
        self.t_begin = t.time()

        # To make batches callable with self.data.next()
        self.data = self.get_infinite_batches(train_loader) 

        # Tensor scalars
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)
        time_start = t.time()
        z_testing = self.get_torch_variable(torch.randn(64, self.n_latent, 1, 1)) # To track generated images

        nb_grad_nan = 0

        for g_iter in range(self.generator_iters):
        
            # D requires grad to compute its derivatives
            for p in self.D.parameters():
                p.requires_grad = True
            print("Critic iter :",self.critic_iter)

            # Following the loss
            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0

            # Train D self.critic_iter times for 1 G step
            for d_iter in range(self.critic_iter):
                self.D.zero_grad() #init

                images = self.data.__next__() #new training images

                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue
                
                # Latent vector
                z = torch.rand((self.batch_size, self.n_latent, 1, 1))

                # Training + latent as torch variables
                images, z = self.get_torch_variable(images), self.get_torch_variable(z)
               
                # Train discriminator

                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, self.n_latent, 1, 1)) # Redondant ?
                fake_images = self.G(z)
                if g_iter > 0:
                    #Noising the images during D training (except at first step)
                    fake_images = self.noiser(fake_images)
              
                # Compute loss on fake images
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()

                # Total D loss
                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake # Without GP
                self.d_optimizer.step() # One step

                # Storing and printing
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
                self.L_loss_D.append(- Wasserstein_D.cpu().item())
                self.L_loss_D_2.append(- Wasserstein_D.cpu().item()/2)
                self.L_loss_D_fake.append(d_loss_fake.cpu().item())
                self.L_loss_D_real.append(-d_loss_real.cpu().item())
                self.L_loss_D_GP.append(d_loss.cpu().item()/2)
            

            # G training 
            # We deactivate gradient on D
            for p in self.D.parameters():
                p.requires_grad = False

            self.G.zero_grad() 
            # compute loss with fake images
            # New latent vectors
            z = self.get_torch_variable(torch.randn(self.batch_size, self.n_latent, 1, 1))
            fake_images = self.G(z)
            
            # Unequalization of the output images
            if self.uneq is not None:
                fake_images_uneq = self.uneq(fake_images)
            
            if g_iter > 0:
                # Noising images for D
                fake_images = self.noiser(fake_images)

            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
                
            print("Sum : ", torch.sum(fake_images_uneq)/self.batch_size) # Check if the sum ~1

            g_loss.backward(mone)            

            print("Weights nan :", np.isnan(self.G.main_module[0].weight.grad.detach().cpu().numpy()).any()) # Check that there is no nan

            if np.isnan(self.G.main_module[0].weight.grad.detach().cpu().numpy()).any():
                print("No G upgrade : nan in gradients, for the ", nb_grad_nan,"th time")
                nb_grad_nan +=1
            else:
                self.g_optimizer.step()

            # Following the training advancement
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            print("Number of nan in gradient :", nb_grad_nan)
            

            # Storing loss
            self.L_loss_G.append(g_loss.cpu().item())

            # Saving intermediate models each 10 000 G iterations
            if (g_iter) % 10000 == 0:
                self.save_model(add_name = "_"+str(g_iter))

            # Saving last model and sampling images every 1000 G iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model() 
                print("Time :",  round(t.time()-time_start), "s, ", round((t.time()-time_start)/60, 1), "min, ", round((t.time()-time_start)/3600, 1), "h")

                # Initialize testing of image generation along the iterations
                if not os.path.exists('training_result_images'+self.name_suite+'/'): #Random images
                    os.makedirs('training_result_images'+self.name_suite+'/')
                if not os.path.exists('training_result_images_same'+self.name_suite+'/'): # Random images with same latent vector
                    os.makedirs('training_result_images_same'+self.name_suite+'/')
                if not os.path.exists('training_result_images_same'+self.name_suite+'_noised/'): # Same with simulated noise
                    os.makedirs('training_result_images_same'+self.name_suite+'_noised/')

                # Unequalize 64 images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(64, self.n_latent, 1, 1))

                samples = self.G(z) # Random images
                samples_testing = self.G(z_testing) # Random images (from the same z each time)
                samples_testing_noised = self.noiser(samples_testing) # Same but noised
                
                samples = samples.data.cpu()[:64]
                samples_testing = samples_testing.data.cpu()[:64]
                samples_testing_noised = samples_testing_noised.data.cpu()[:64]
                
                grid = utils.make_grid(samples)
                grid_testing = utils.make_grid(samples_testing)
                grid_testing_noised = utils.make_grid(samples_testing_noised)
                
                utils.save_image(grid, 'training_result_images'+self.name_suite+'/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))
                utils.save_image(grid_testing, 'training_result_images_same'+self.name_suite+'/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))
                utils.save_image(grid_testing_noised, 'training_result_images_same'+self.name_suite+'_noised/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))
    

                # Loss plotting
                if not os.path.exists('training_loss'+self.name_suite+'/'):
                    os.makedirs('training_loss'+self.name_suite+'/')
                try:
                    self.plot_losses(save=True, path_save='training_loss'+self.name_suite+'/iter'+str(g_iter)+'.pdf')
                except:
                    print("Pb with iteration", g_iter)                          

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # At the end : save the final trained parameters
        self.save_model()


    # (Obsolete)
    def evaluate(self, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, self.n_latent, 1, 1))
        samples = self.G(z)
        # samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'wgan_model_image.png'.")
        utils.save_image(grid, 'wgan_model_image.png')


    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
    
    # (Obsolete)
    def real_images(self, images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 64, 64)[:self.number_of_images])


    # (Obsolete)
    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample[0])
        return generated_images

    # (Obsolete ?)
    def to_np(self, x):
        return x.data.cpu().numpy()


    def save_model(self, add_name=""):
        torch.save(self.G.state_dict(), './generator'+self.name_suite+add_name+'.pkl')
        torch.save(self.D.state_dict(), './discriminator'+self.name_suite+add_name+'.pkl')
        print('Models save to ./generator'+self.name_suite+add_name+'.pkl & ./discriminator'+self.name_suite+add_name+'.pkl')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))
    
    # If I want to load only G (to not waste RAM when we use it after training)
    def load_model_G(self, G_model_filename):
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))

    # Manage batches
    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    # (Obsolete)
    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, self.n_latent, 1, 1)
        z1 = torch.randn(1, self.n_latent, 1, 1)
        z2 = torch.randn(1, self.n_latent, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")


    # Plot losses 
    def plot_losses(self, save=False, path_save=""):
        n_it_G = len(self.L_loss_G)
        n_it_D = int(len(self.L_loss_D) * (self.critic_iter + 1)/(self.critic_iter))
        x_G = [(self.critic_iter + 1)*(i+1) for i in range(n_it_G)]
        #x_D = [i+1 for i in range(n_it_D) if i+1 not in x_G]

        fig2 = plt.figure(2)

        to_plot = self.L_loss_D_fake[self.critic_iter-1::self.critic_iter][:len(x_G)]
        plt.plot(x_G[:len(to_plot)], to_plot, "--", label="Critic - fake", alpha=0.5)
        to_plot = self.L_loss_D_real[self.critic_iter-1::self.critic_iter][:len(x_G)]
        plt.plot(x_G[:len(to_plot)], to_plot, "--", label="Critic - real", alpha=0.5)
        to_plot = self.L_loss_G[:len(x_G)]
        plt.plot(x_G[:len(to_plot)], to_plot, label="Generator")
        to_plot = self.L_loss_D_2[self.critic_iter-1::self.critic_iter][:len(x_G)]
        plt.plot(x_G[:len(to_plot)], to_plot, label="Critic/2")
        to_plot = self.L_loss_D_GP[self.critic_iter-1::self.critic_iter][:len(x_G)]
        plt.plot(x_G[:len(to_plot)], to_plot, alpha=0.7, label="Critic GP/2")
        if self.L_loss_lambda_HF != []:
            to_plot = self.L_loss_lambda_HF[:len(x_G)]
            plt.plot(x_G[:len(to_plot)], to_plot, label="HF loss")
        #if self.L_loss_lambda_sum1 != []:
        #    to_plot = self.L_loss_lambda_sum1[:len(x_G)]
        #    plt.plot(x_G[:len(to_plot)], to_plot, label="Sum1 loss")
        plt.legend()

        if save:
            plt.savefig(path_save)
        else:
            plt.show()
        plt.close("all")
