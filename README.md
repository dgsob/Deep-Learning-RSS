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
- Deep Learning in Python: 
    - [JAX](https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research) and [Equinox](https://docs.kidger.site/equinox/all-of-equinox) - for hardware acceleration, optimizations, and DeepMind ecosystem familiarity
    - PyTorch [Course](https://www.youtube.com/watch?v=V_xro1bcAuA) - for quicker prototyping, more vibe-coding friendly

### Theory Concepts:
1. Mathematical Background (4.3, 5.4, 5.5, 5.9)
2. Feedforward Networks (6.1-4) and Back-Propagation (6.5)
3. Regularization for DL (7.1, 7.3-5 (skip 7.5.1), 7.7-8, 7.10-13)
4. Optimization (8.1-5 (Skip 8.3.3))
5. Convolutional NN (9.1-9)
6. Recurrent & Recursive Nets (10.1-10 (skip 10.8))

### Notes

#### Ad. 2: Feedforward Networks (6.1-4) and Back-Propagation (6.5)
##### Losses
* cross-entropy between the training data and the model distribution (negative log-likelihood $\rightarrow$ MLE)
```
logits = jnp.array([[1.2, -0.8, -0.5], [0.9, -1.2, 1.1]])
labels = jnp.array([0, 1])

# Softmax
exp_logits = jnp.exp(logits - jnp.max(logits, axis=1, keepdims=True))
softmax_probs = exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)

# Cross-entropy loss
loss = -jnp.log(softmax_probs[jnp.arange(logits.shape[0]), labels])
print(loss)  # Should print ~[0.2761, 2.9518]
```
* MSE, MAE
#### Ad. 3: Regularization in DL (7.1, 7.3-5 (skip 7.5.1), 7.7-8, 7.10-13)
##### L1 and L2 Regularization (Weight Decay)
* What it does: adds a penalty to the loss function based on the size of the model's weights.
* Idea: L1 encourages sparsity by driving some weights to zero, while L2 pushes all weights toward smaller values, reducing model complexity.
* Pros:
    * Simple to implement and works with most optimization algorithms.
    * L1 automatically selects important features by zeroing out others.
    * L2 is differentiable $\rightarrow$ compatible with gradient-based optimization.
    * L2 ensures smooth weight adjustments $\rightarrow$ helps gradient-based optimization.
* Cons:
    * Requires trial-and-error tuning of the penalty strength.
    * May not help with overfitting in very deep or complex models.

##### Early Stopping
* What it does: halts training before overfitting occurs.
* Idea: monitor performance on a validation set and stop training when it begins to degrade, preserving generalization.
* Pros:
    * Easy to apply to any model without need to alter its architecture.
    * Acts as a natural constraint on model complexity by capping training iterations.
* Cons:
    * Needs a separate validation set.
    * Timing the stop can be tricky (too early underfits, too late overfits).

##### Dropout
* What it does: randomly deactivates units during training.
* Idea: by dropping a fraction of neurons in each pass, it prevents units from co-adapting too much, mimicking an ensemble of networks.
* Pros:
    * Reduces feature co-dependency, especially in deep networks.
    * Mimics training multiple models at a fraction of the cost.
* Cons:
    * Computationally expensive due to repeated random sampling.
    * Dropout rate needs careful tuning to avoid underfitting.

##### Data Augmentation
* What it does: artificially expands the training dataset.
* Idea: apply transformations (e.g., rotations, flips) to existing data to create new training examples, improving model robustness.
* Pros:
    * Increases effective dataset size $\rightarrow$ good for limited data scenarios.
    * Increases variations in the dataset.
    * Can be cheap if transformations are applied during training.
* Cons:
    * Not suitable for all data types (e.g., tabular).
    * Requires domain expertise to design transformations.

##### Batch Normalization
* What it does: normalizes layer inputs during training.
* Idea: normalize the inputs to each layer to reduce internal covariate shift, stabilizing and regularizing the learning process.
* Pros:
    * Speeds up training by supporting higher learning rates.
    * Eases gradient flow (assumption: potentially reduces issues like vanishing gradients).
* Cons:
    * Computationally expensive due to extra calculations per layer (overhead).
    * Less impactful in shallow or small networks.

##### Ensemble Methods
* What it does: combines multiple models' predictions.
* Idea: train several models and aggregate their outputs to reduce variance and improve generalization.
* Pros:
    * Lowers prediction variance by blending different model strengths.
    * Pairs well with other techniques for added benefits.
* Cons:
    * Computationally expensive.
    * Requires training and managing multiple models.

##### Adversarial Training
* What it does: incorporates adversarial examples into the training process.
* Idea: expose the model to perturbed inputs (adversarial examples) to enhance robustness and generalization.
* Pros:
    * Improves resilience to adversarial attacks.
    * Acts like data augmentation by introducing challenging examples.
* Cons:
    * Computationally expensive due to generating perturbations (overhead).
    * May weaken performance on unperturbed, standard data.
    * Requires domain knowledge and careful design to mitigate possible negative side-effects.

##### Parameter Sharing
* What it does: reuses parameters across model components.
* Idea: share weights (e.g., in convolutional layers) to reduce the number of unique parameters, constraining the model and aiding generalization.
* Pros:
    * Shrinks model size, making it more efficient to train and run.
    * Leverages data structure (e.g., spatial patterns) for better learning.
* Cons:
    * Limited to architectures where sharing fits (e.g., not fully connected layers).
    * May restrict the model’s flexibility if over-applied.

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