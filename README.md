

![guygradlogo](https://github.com/GuyE2718/GuyGrad/assets/121691820/04f0dfd7-8380-4c48-97d0-7d5b2d0c9215)

GuyGrad is a C++ implementation of [Micrograd](https://github.com/karpathy/micrograd), originally created by Andrej Karpathy. This was a fun little project, took about a day to complete the core but then got stuck debbuging a problem with the shared pointers. I loved Karpathy's elegant Micrograd library and wanted to learn about it more, and I think implementing it from scratch in C++ is the best way to learn.

GuyGrad is a scalar autograd engine, which means automatic gradient computation. Essentially, it's a framework that implements backpropagation. Backpropagation is an algorithm designed to estimate the gradient of a given loss function with respect to the weights of a neural network. By tuning these weights to minimize the loss function, the network's accuracy can be improved.

for more information about how micrograd works you can checkout [Karpathy's video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) explaining how it works

I used the example in Karpathy's lecture to demonstrate how it works: 
```cpp
auto x1 = std::make_shared<Value>(2.0);
auto x2 = std::make_shared<Value>(0.0);

auto w1 = std::make_shared<Value>(-3.0);
auto w2 = std::make_shared<Value>(1.0);

auto b = std::make_shared<Value>(6.8813735870195432);

auto x1w1 = x1 * w1;
auto x2w2 = x2 * w2;
auto x1w1x2w2 = x1w1 + x2w2;

auto n = x1w1x2w2 + b;

auto o = n->tanh();

o->backward();

    

std::cout << "w1 grad: " << w1->grad << std::endl;
std::cout << "w2 grad: " << w2->grad << std::endl;
```
```
output: 
w1 grad: 1
w2 grad: 0
```
![sch](https://github.com/GuyE2718/GuyGrad/assets/121691820/d44c86c7-cbb4-49af-bbc4-5f646c94af91)
