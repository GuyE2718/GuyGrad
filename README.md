

![guygradlogo](https://github.com/GuyE2718/GuyGrad/assets/121691820/04f0dfd7-8380-4c48-97d0-7d5b2d0c9215)

GuyGrad is a C++ implementation of [Micrograd](https://github.com/karpathy/micrograd), originally created by Andrej Karpathy. This was a fun little project, took about a day to complete the core but then got stuck debbuging a problem with the shared pointers. I loved Karpathy's elegant Micrograd library and wanted to learn about it more, and I think implementing it from scratch in C++ is the best way to learn.

GuyGrad is a scalar autograd engine, which means automatic gradient computation. Essentially, it's a framework that implements backpropagation. Backpropagation is an algorithm designed to estimate the gradient of a given loss function with respect to the weights of a neural network. By tuning these weights to minimize the loss function, the network's accuracy can be improved.

for more information about how micrograd works you can checkout [Karpathy's video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) explaining how it works

