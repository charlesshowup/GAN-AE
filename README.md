# GAN-AE
This object use AE to help GAN's elevate training of discriminator 

__Table of Contents__
We present a general learning framework that combines a Deep Convolutional GAN(DCGAN) with a Conditional Variational Auto-Encoder (DCVAE). First, we find a solution that Wasserstein-DCGANs solve the gradient vanishing problem to a certain extent, but the generated results are usually blurry. Second, we adopt a VAE to remove useless features from real images, reducing noises of images. We further experiment with natural images of faces, flowers that the proposed models are capable of generating sharp and diverse samples. 

## Why we made Wasserstein-DCGAN + Conv-VAE
VAE model is often used to process noise reduction of music, videos and pictures because of its ability of throwing unnecessary features. 
Unfortunately, when the picture through the VAE model, we often get a blurry picture though our model is well-done. Besides, when we input a random noise to the decoder, we can still get a unsharp picture which conform to the distribution of real images.
Following, we summarize the pros and cons about VAE model.
Pros: 
- Principled method to generative models
- Allow q(z|x) to infer other tasks

Cons: 
- Samples blurrier than GAN and low quality
- Maximize lower bound of likelihood
Architecture of GAN
Nevertheless, everything is two-faced. Though GAN can help us create new things, it still has disadvantages. GAN’s pros and cons are following.
Pros:
- Beautiflu and state of the art samples
Cons:
- Hard to train because of its instability
- Can’t solve inference queries
Therefore, we image that utilizing the advantages of VAE and GAN to solve the cons between them. In the article, we use VAE model to decrease the influence of noise in the photos, then use GAN to create new pictures. In this case, GAN will accept fewer insignificant features if we input the pictures which were processed by VAE model. Such process not only depends on the ability of VAE, but also decrease the risk of collapse of training GAN.
> To forestall great economic losses, it is significant to create a method to consumption of generate virtual face. We optimized VAE model and use the picture which output from GAN as the input of VAE. Then, we can generate and modify some blurry photos.

## Experiments
### Data
- __Prepare dataset__
We selected two datasets, [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md), [CelebA](https://www.google.com/search?client=safari&rls=en&q=CelebA&ie=UTF-8&oe=UTF-8). In FashionMNIST which includes several types of clothes, it has about 60,000 samples. Besides, CelebA is a dataset which contains 202,599 face images

- __Convolutional Neural Network (CNN)__
We combine CNN with VAE and GAN, meaning that the result we get can be sharper and more reliable.
- __Variational Autoencoder (VAE)__
We set up a framework which contains 7 CNNs and 2 resnets both in encoder and decoder. 
- __Generative Adversarial Network (GAN)__
We set up several generative adversarial networks include Wasserstein-DCGAN model.

### Combine GAN with VAE
- __Baseline__
As we decided to process Wasserstein-DCGAN and VAE, we selected Linear Autoencoder, Linear Variational Autoencoder, Linear GAN, Linear WGAN as our baselines. Those models illustrate the previous preparing work we had done.
- __Different combination of model__
In order to verify the best combination of GAN-VAE, we prepare different types of combination which contain a type of autoencoder and a type of GAN.
