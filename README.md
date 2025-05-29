## Repo to keep track of DL progress

### Goal:
To be able to:
* Explain basic & modern concepts
* Select appropiate training & validation methods
* Compare pros & cons of different architectures
* Compare pros & cons of different training, regularization & validation schemes
* Perform problem diagnosis & troubleshooting

### Main resources:
- The MT7042-HT24 course-book: ["Deep Learning" by Goodfellow et al., MIT Press, 2016](https://www.deeplearningbook.org/)
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/), the "proto-book"
- Julia Deep Learning library [Lux.jl](https://lux.csail.mit.edu/stable/)

### Theory Concepts:
1. Mathematical Background (4.3, 5.4, 5.5, 5.9)
2. Feedforward Networks (6.1-4) and Back-Propagation (6.5)
3. Regularization for DL (7.1, 7.3-5 (skip 7.5.1), 7.7-8, 7.10-12)
4. Optimization (8.1-5 (Skip 8.3.3))
5. Convolutional NN (9.1-9)
6. Recurrent & Recursive Nets (10.1-10 (skip 10.8))

### Notes

Ad. 4: Optimization in DL
1. Early stopping.
2. A sampling-based estimate of a gradient is a win-win.
3. Minibatch methods are more optimal.
4. Batch size: 32-256, with 16 sometimes attempted for large models. 
    * smaller batches $\rightarrow$ add noise $\rightarrow$ regularization effect $\rightarrow$ better generalization error
    * smaller batches $\rightarrow$ add noise $\rightarrow$ high variance in the estimate of the gradient $\rightarrow$ require reduced $\alpha$ to maintain stability $\rightarrow$ slow training
    * smaller batches $\rightarrow$ it takes more steps to observe the entire training set $\rightarrow$ slow training
5. Suffle the examples before training so that the minibatches are selected randomly, without correlated subsequent samples.
6. Saddle points are more common for high dimensional loss functions than local minima. Additinally, these local minima are more likely to be low-cost, making saddle points way more likely to be a hgh-cost critical point in such high-dimensional spaces. Empirically gradient descent (GD) seems to be able to escape saddle points tho.
7. GD just moves downhill. Second-order methods on the other hand are explicitly designed to seek a critical point. Thus saddle points constitute much bigger of a problem for them. Besides, they don't scale to large neural networks at the time.
8. We can avoid cliffs in traditional GD by using gradient clipping heuristic - it reduces step size.
9.