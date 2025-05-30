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
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/)$^*$, specifically the "[proto-book](https://arxiv.org/pdf/2104.13478)"
- Julia Deep Learning library [Lux.jl](https://lux.csail.mit.edu/stable/)

### Theory Concepts:
1. Mathematical Background (4.3, 5.4, 5.5, 5.9)
2. Feedforward Networks (6.1-4) and Back-Propagation (6.5)
3. Regularization for DL (7.1, 7.3-5 (skip 7.5.1), 7.7-8, 7.10-12)
4. Optimization (8.1-5 (Skip 8.3.3))
5. Convolutional NN (9.1-9)
6. Recurrent & Recursive Nets (10.1-10 (skip 10.8))

### Notes

#### Ad. 4: Optimization in DL (8.1-5 (Skip 8.3.3))  
##### 8.1 "How learning differrs from pure optimization"
1. Early stopping.
2. A sampling-based estimate of a gradient is a win-win.
3. Minibatch methods are more optimal.
4. Batch size: 32-256, with 16 sometimes attempted for large models. 
    * smaller batches $\rightarrow$ add noise $\rightarrow$ regularization effect $\rightarrow$ better generalization error
    * smaller batches $\rightarrow$ add noise $\rightarrow$ high variance in the estimate of the gradient $\rightarrow$ require reduced $\alpha$ to maintain stability $\rightarrow$ slow training
    * smaller batches $\rightarrow$ it takes more steps to observe the entire training set $\rightarrow$ slow training
5. Shuffle the examples before training so that the minibatches are selected randomly, without correlated subsequent samples.
##### 8.2 "Challanges in neural network optimization"
1. **Saddle points** are more common for high dimensional loss functions than local minima. Additinally, these local minima are more likely to be low-cost, making saddle points way more likely to be a hgh-cost critical point in such high-dimensional spaces. Empirically gradient descent (GD) seems to be able to escape saddle points tho.
2. Second-order methods, unlike GD, are explicitly designed to seek a critical point. Thus saddle points constitute much bigger of a problem for them. Besides, they don't scale to large neural networks at the time.
3. We can avoid **cliffs** in traditional GD by using gradient clipping heuristic - it reduces step size.
4. Very deep NNs can experience **vanishing and exploding gradients** problem, especially RNN. Cliffs are an example of exploding gradients.
5. All algorithms like GD are focused on - and all above issues relate to - making small local moves. The path found in this way doesn't necessarily correspond to a good solution. The research focuses on better initialization, rather than developing algorithms that use nonlocal moves and global structure.
##### 8.3 "Basic algorithms"
1. SGD.
2. Momentum.
##### 8.4 "Parameter initialization strategies"
1. We don't know much about optimization in DL, all relies on trial and error, including initialization. 
2. The only thing known for sure is that the initial parameters need to break symmetry between different units - if two different units with the same activation function are connected to the same inputs, then these units must have different initial parameters.  
3. Weights should be initialized randomly, how does not matter much, their scale matter.From optimization pov weights should be as large as possible, from regularization pov, they should be near 0. Different scaling heuristic schemes exist. They all suck. Refer to the book for a "good rule of thumb". 
4. Biases should be initialized in a way that's compatibile with weights initialization scheme, usually 0 is ok.
##### 8.5 "Algorithms with adaptive learning rates"
1. AdaGrad
2. RMSProp
3. Adam (adaptive moments)
##### Algorithms actively in use: SGD, SGD with momentum, RMSProp, RMSProp with momentum, AdaDelta, Adam.












                     
---
$^*$ The people in the pictures on this webpage are Felix Klein and Emmy Noether. The authors of the book "make a modest attempt to apply the Erlangen Programme mindset to the domain of deep learning, with the ultimate goal of obtaining a systematisation of this field and ‘connecting the dots’." They "call this geometrisation attempt ‘Geometric Deep Learning’, and true to the spirit of Felix Klein, propose to derive different inductive biases and network architectures implementing them from first principles of symmetry and invariance."