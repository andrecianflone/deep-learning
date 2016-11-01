see [this](https://blog.keras.io/building-autoencoders-in-keras.html) blogpost. The post is also a good tutorial for Autoencoders in Keras.

### What are autoencoders good for?
- Not very good at data compression, much better algorithms out there
- Rarely used in practice in original form.
- Mostly used for data denoising and dimensionality reduction for visualization

### Examples
- Autoencode MNIST from 784, down to 32, and then back. When applying to the test set, reconstruction from low 32 dimension looks like original but blurry.
- Can add a sparsity contraint on the activity of the hidden representations, type of regularizer
- Deep autoencoder. Instead of layers input/hidden/output [784 32 784], we can try [784 128 64 32 64 128 784]. Note the progressive compression/decompression. However, only minute improvement in performance.
- Convolution autoencoder: Much better than vanilla autoencoder due to the "higher entropic capacity of the encoded representation, 128 dimensions vs. 32 previously".
- Denoising: Train an autoencoder to map noisy images to clean images. We can easily do this by adding Gaussian noise.
- Variational autoencoder: Generative model, you learn parameters of a probability distribution modeling your data. You can then sample this distribution.
