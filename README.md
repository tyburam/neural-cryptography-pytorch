# neural-cryptography-pytorch
PyTorch reimplementation of [neural-cryptography-tensorflow](https://github.com/ankeshanand/neural-cryptography-tensorflow)

# Adversarial Neural Cryptography in [PyTorch](https://github.com/pytorch/pytorch)

A PyTorch implementation of Google Brain's paper ([Learning to Protect Communications with Adversarial Neural Cryptography.](https://arxiv.org/pdf/1610.06918v1.pdf))

Two Neural Networks, Alice and Bob learn to communicate secretly with each other, in presence of an adversary Eve.

![Setup](assets/diagram.png)

## Problems 
Currently it fails in forward method of CommunicateNet for first convolutional layer with some nasty memory error in C++ code
   
     RuntimeError: $ Torch: invalid memory size -- maybe an overflow? at /Users/administrator/nightlies/pytorch-1.0.0/wheel_build_dirs/wheel_3.7/pytorch/aten/src/TH/THGeneral.cpp:188

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

