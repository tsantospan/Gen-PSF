# Gen-PSF
# Parametrization of PSF based on Generative neural networks


This repo presents the principal codes used in the paper *Data-driven PSF parametrization with generative neural networks*. The project consists in training a generative neural network on images of real measured noisy PSF. Once trained, the generator is used as a PSF parametrization without noise. The training follows the classical framework of Wasserstein GAN with Gradient Penalty, with some additions specific to this project, regarding the problematics of equalization and noise addition that are discussed in the paper.

This repo mostly focuses on the training (Section 3 of the article) and the application to PSF denoising (Section 5 of the article).


## Root folders

- *data* : contains raw data. Some of it must be downloaded with a specific script.
- *models* : stores the trained models.
- *code* : contains the main codes, for training and using the trained models
- *results* : we provide here the results of PSF denoising (described in the article)

More details will be provided.


## Downloading raw data

Part of the raw data is directly stored in the repo, because of the small volumes it taekes . It is the case for the statistics of the PSF (used for noise simulation and equalization), examples of real PSF and their denoised counterparts (see the article).

The rest of raw data (training and test sets, already trained generators that are used in the article) must be downloaded with the script.

## Folder **data** 
### Subfolder *sets*
The training and test sets are stored in this subfolder. They must be downloaded with the script. Each name corresponds to the filters and the years (eg H23_2016_2019 for the filters H2 and H3 with PSF measured between 2016 and 2019). In each one of these folders, we separate again in two folders depending on the equalization (log or white). We then finally separate between training and test set. 

To summarize, here is the structure : 
- filter and year
  - equalization
    - training / test

For the code to be working well, the training set path (here, the path to the folder *training)* provided in the code must contain a subfolder (named here *train*) where are stored the training images.

The training images consist in 128 $\times$ 128 pixels images in png format.



### Subfolder *statistics*
This folder contains the statistics that are used for noise simulation.

- Read-out noise : we provide two examples of read-out noise maps (files begininning with *ron*)
- Mean and standard deviation PSF : it is the mean and standard deviation (computed at each pixel) of the PSF of a given training set. They are used for the whitening. The name begings with MEAN or STD, followed by information about the training set (filters and years).
- Sum and number of frames : we also provide the statistics about the PSF sums (before normalization) and their number of frames. The files names begin with alphas_Ns, followed by the corredponding training set (filters and years)

### Subfolder *observations_PSF*
We provide the PSF that are used for the application with PSF denoising, in Section 5. The given PSF corresponds to observations of  PDS 70, SAO 206462, HIP 72192 and HIP 80019.

## Folder **models**
We provide examples of trained generators that were used to obtain results presented in the article, in Section 5.
All four generators provided were trained with the K12 training set. Two of them were trained on whitened images, with 10 latent dimensions, and the two others on log-equalized images, with 20 latent dimensions. For each training configuration, we provide two different generators trained at different times, identified by the year of the training (2024 or 2025).

## Folder **code**
The first subfolder *training* contains two files : 
- *wgan_gradient_penalty.py*, which defines the class for the WGAN
- *training_wgan.py* which is used to train the neural network. This script has three possible arguments : 
  - nz : the latent dimensionality required
  - training_s : the training set (1617 for H2 and H3 filters from years 2016 and 2017, 1619 for H2 and H3 filters from years 2016 to 2019,1719 for K1 and K2 filters from years 2017 to 2019)
  - mode_eq : the equalization (white or log)
  
  Example use : `python3 training_wgan.py --nz 20 --training_s 1619 --mode_eq white`

The second subfolder contains a notebook to illustrate the denoising with the generator. It works with PSF observed with K1 and K2 filters, with one or two frames.

## Folder **results**
We provide here already computed results of PSF denoising. They are the denoised PSF that are presented in the article in Section 5. There is one folder for each corresponding obersved object (PDS 70, SAO 206462, HIP 72192 and HIP 80019). In each folder there are two subfolders corresponding to the denoising obtained with a generator trained with either whitened PSF or log-equalized PSF. In each subfolder, we show the PSF denoised with either the generator trained in 2024 or 2025, or a zoom on the 64 $\times$ 64 pixels at the center.
