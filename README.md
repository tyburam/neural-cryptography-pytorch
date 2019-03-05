# neural-cryptography-pytorch
PyTorch reimplementation of [neural-cryptography-tensorflow](https://github.com/ankeshanand/neural-cryptography-tensorflow)

# Adversarial Neural Cryptography in [PyTorch](https://github.com/pytorch/pytorch)

A PyTorch implementation of Google Brain's paper ([Learning to Protect Communications with Adversarial Neural Cryptography.](https://arxiv.org/pdf/1610.06918v1.pdf))

Two Neural Networks, Alice and Bob learn to communicate secretly with each other, in presence of an adversary Eve.

![Setup](assets/diagram.png)

## Problems 
Currently it fails in training method
   
    $ RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.

## Useful information
PyTorch has a tiny little difference in convolution layer when compared with TensorFlow.  
TF treats convolution input tensor as [batch_size, len, channels] while PyTorch way is [batch_size, channels, len]
## Pre-requisites

* PyTorch
* Numpy

## Usage 
First, ensure you have the dependencies installed.

    $ pip install -r requirements.txt

To train the neural networks, run the `main.py` script.

    $ python main.py --msg-len 32 --epochs 50
    
## Attribution / Thanks

* ankeshanand's implementation in TensorFlow [neural-cryptography-tensorflow](https://github.com/ankeshanand/neural-cryptography-tensorflow)

## License

MIT

