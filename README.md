Torch implementation of

(i) Frank-Wolfe algorithm
(ii) Block-coordinate Frank-Wolfe algorithm
(iii) Minibatch version of BCFW

BCFW takes up a lot of memory, since w vector is stored for each sample.
The execution for BCFW does not fit in my computer (16GB) for 50k training size of CIFAR-10
Hence, minibatch version is implemented.

Implementation of each of the three algorithms is in /solver with respective names.

The way to call them is illustrated for the example of cifar-10 in cifar.lua

A few points to note:

1. I found that if we remove line search for BCFW and minibatch version, the primal, dual and gap values
have a large magnitude. The step size values are much larger as compared to the line search case. I got same
results on using Matlab code without line search.

2. I am yet to do a comparison of the three algorithms in terms of dual value vs time. When I gave the run for it,
it terminates at some point because of gnuplot not having pdf functionality. This is strange, since most times plotting
works fine.

In /helpers/helperFn.lua, these are implemented for cifar-10 and mnist:

(i) feature vector computation
(ii) loss function
(iii) max oracle
(iv) primal function, dual function, duality gap

