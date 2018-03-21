# DeepfakeGan
Using GANS to generate images that are then used for deep fakes

A generative model is a system that performs the difficult task of generating novel data points given the current data distribution, and requires enormous amounts of data to train. A discriminative model on the other hand acts as a classifier, giving us the probability of an input belonging to a particular class. A GAN (Generative Adversarial Network) is a very popular machine learning technique created just two years ago that has found uses in almost every field. Applications include drug discovery by generating sample drug candidates, image generation from a text description, applying color or transfering a style to a drawing or an image. It has 2 components which are trained at the same time as adversaries in a minimax game: a generator that attempts to transform samples drawn from a prior distribution to samples from a complex data distribution with much higher dimensionality and a discriminator that decides whether the given sample is real or generated. The generator is trained to fool the discriminator by creating realistic images, and the discriminator is trained not to be fooled by the generator. However, GANS are notorious for being difficult to train due to the minimax optimization and the training might be quite unstable with vanishing gradients, mode collapse etc. Traditionally, GANS use CNNs but since Capsule networks were recently shown to outperform CNNs, we would like to attempt designing a GAN using Capsule networks and employ several techniques devised to stabilize and make training more robust. Internal data representation of a CNN does not capture important spatial hierarchies between simple and complex objects. Capsules on the other hand are a groups of neurons that are locally invariant and learn to recognize visual entities and output activation vectors that represent both the presence of those entities and their properties relevant to the visual task.

## Potential Datasets - MNIST, CIFAR-10, Celebrities faces dataset

## Evaluation: 
We would have to qualitatively compare images generated randomly using both a regular GAN and our model for the lack of a proper metric (Human evaluation). Quantitatively we can perhaps try some classification task using the generated images. 

## References:

* Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio.Generative Adversarial Nets. In Advances in neural information processing systems, pages 2672–2680,2014
* Geoffrey Hinton, Sara Sabour, Nicholas Frosst, MATRIX CAPSULES WITH EM ROUTING.ICLR 2018
* S. Sabour, N. Frosst, and G. E. Hinton. Dynamic routing between capsules. In Advances in Neural Information Processing Systems, pages 3859–3869, 2017.
* T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, X. Chen, and X. Chen. Improved techniques for training gans. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors,Advances in Neural Information Processing Systems 29, pages 2234–2242. Curran Associates, Inc., 2016
