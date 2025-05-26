using CSV
using DataFrames
using StatsBase
using Random

function inspect_data(data)
    selected_cols = vcat(1:5, 55)

    println("Head (first 5 rows, selected columns):")
    println(first(data[:, selected_cols], 5))
    
    println("Summary:")
    println(describe(data))
end

function inspect_class_distribution(data)
    class_counts = combine(groupby(data, :Cover_Type), nrow => :count)
    class_counts.proportion = class_counts.count / sum(class_counts.count)

    println("Class distribution:")
    println(class_counts)
end

function split_dataset(df::DataFrame, training_portion=0.8, validation_portion=0.1, seed=77)
    Random.seed!(seed)
    n = nrow(df)
    indices = shuffle(1:n)

    train_end = Int(floor(training_portion * n))
    val_end = train_end + Int(floor(validation_portion * n))

    train_set = df[indices[1:train_end], :]
    val_set = df[indices[train_end+1:val_end], :]
    test_set = df[indices[val_end+1:end], :]

    return train_set, val_set, test_set
end

function standardize(data::Vector{<:Integer}, mean, std_dev)
    if std_dev == 0
        @warn "Standard deviation is 0. Skipping standardization."
        return Float64.(data)
    end

    return (data .- mean) ./ std_dev  # z = (x - μ) / σ
end

function get_statistics(data::Vector{<:Integer})
    n = length(data)

    mean = sum(data) / n  # μ = Σx_i / n               
    std_dev = sqrt(sum((item - mean)^2 for item in data) / n)  # σ = √(Σ(x_i - μ)² / n)

    return mean, std_dev
end

#= Task 1 (2 p): 
Load The Forest Covertype Dataset from the CSV file
covertype.csv found in the projects directory. Make sure to read and
understand the description before proceeding. 
=#

#= Task 2 (3 p): 
Check the class distribution of the data. Is class imba-
lance an issue? If so, explain why it could negatively affect the network
performance. 
=#

#= The classess are imbalanced. It means the majority classes contribute
more to the loss, and an overall high accuracy can be achieved by simply
predicting the majority class, instead of catching patterns actually, 
and universally, useful for prediction. 
=#

#= Task 3 (5 p): 
Standardize the dataset by writing your own function
without using any external libraries. 
=#

#= Task 4 (5 p): 
Explain why standardization is important in the context of
training a neural network. Should this step be performed before or after
splitting the data? Justify your answer. 
=#

#= Standardization puts all features on a comparable scale. This is important
for neural networks because their pattern recognition relies on gradient-based
backpropagation algorithm, and gradient is sensitive to the scale of input features - larger 
values produce larger gradients. In theory NN could learn to adjust the weight of respective 
feature with scale that's too large, but in reality it's a huge obstacle and there's no point 
is spending the NN's time to learn it, when it can be easily removed before training. 

Standardization should be performed after splitting the data to prevent data leakage, 
i.e., influencing the test set with information from the training set. 
In practice, a model deployed on new data only has access to the training data’s statistics. 
Standardizing with test data included simulates knowing the future, which is unrealistic.
=#

#= Task 5 (3 p): 
Split the dataset into training (80%), validation (10%), and test (10%) sets. 
=#

#= Task 6 (30 p): Implement the training of a neural network using the
package of your choice. Motivate your choice of depth and width, activa-
tion function, cost function, output function, parameter initialization, and
training algorithm. A sentence or two is sufficient for each justification.
During training, monitor the training error as well as the validation error.
=#

#= Design chocies:
    > depth and width: 2, 16 <----------------------------------------------------------------------------------------------------
    Based on DLbook Chapter 6 I guess that [more layers + fewer units ---> better generalization].
    At the same time my intuition is that the most shalow network we can get away with is best.
    We'll start with 2 layers and 16 neurons, and see how it goes.
    > activation function: ReLU <-------------------------------------------------------------------------------------------------
    I like ReLU. It's simple, cool, and kinda genius. We will see if this project uncovers any potential
    needs for modifications/alternatives.
    > output function: 7 logits with softmax <------------------------------------------------------------------------------------
    We need to assign a sample to one of 7 classes. It is a multi-class classification problem.
    We should aim to output a probability distribution over all classes.
    This would allow us to quantify uncertainty with directly interpretable numbers. This allows us to 
    make an informed decision (like pick class with the highest probability) based on measurable premises,
    which can be traced back and used to first evaluate, and then improve the model's performance. 
    In order to achieve that, we will use 7 neurons in the output layer. Each will then output a raw score
    for each class, a logit. Then we will appply softmax to these logits  - a straightforward way to make 
    them add up to 1. Then the output is the max over the distribution. 
    > loss function: Weighted cross-entropy loss <---------------------------------------------------------------------------------
    We need a way to quantify error between the predicted class by the output function, and the true label.
    We also have imbalanced classes, and need to deal with that as well.
    Based on our assumption that softmax outpus are good, we need something that is compatible with softmax 
    outputs and supports gradient-based optimization. Since the output layer uses softmax to produce probabilities, 
    the loss should interpret these as a categorical distribution. Given the true label and input features, 
    MLE framework can be uset to predict model parameters θ (wieghts, biases) that would result in these features 
    being observed for this label. This, simplifying, gives us a direct comparison with our output function. 
    We will use log-likelihood, becasue addition (log is monotonic) is more numerically stable than multiplication 
    (can go to extremely small numbers), and we will spcefically minimize negative log-likelihood, becasue this means 
    going from something positive to 0, which is the most stable, intutive and reliable way to optimize something, 
    alligning with the idea of gradient descend: θ <- θ - η * ∇NLL. We will also add weights to account for class 
    imbalance to each of this output, making the underrepresented classes artificially more visible.
    > initialization: He for hidden layers, Glorot for output <---------------------------------------------------------------------
    Most natural phenomena follow some sort of distribution resembling a Gaussian, in one way or another. 
    Gaussian-based random wieghts initialization should be a good starting point. He and Glorot were 
    suggested by Grok, they both allign with my assumption, only adjusting variance in a way to fit ReLU and softamx. 
    In a more serious problem we would use similar approach to actually pretrain initialization values, and bias them 
    towards the right values on the start, or use previous experience from similar setting, like we often do in RL. 
    > training algorithm: Adam <-----------------------------------------------------------------------------------------------------
    We already implicitly made the decision for the algorithm to be based on gradient descend + backpropagation. 
    We mentioned θ <- θ - η * ∇NLL, where η is some learning rate. However, the learning rate should be adjusted 
    dynamically to address the exploration-exploitation trade-off. We should also do mini-batch updates, as updating 
    everything at once would be slow. There must be a way to point these scattered gradients to the same general direction, 
    ignoring local trends. We would do stochastic updates to escape these local trends if necessary. Additionally, we want 
    to somehow remember the direction we are going to, this can be achieved with an exponentially weighted moving average 
    of past gradients, aka momentum. It results in smoothing out the oscillations and guiding the model toward the minimum 
    of the loss function, but of course the direction is more biased to the past experience - if the course was good from
    the beginning, we will arive at the destionation faster. If not, it will take us way longer to realize that. 
    However in classification problems, it seems like the benefits outwieght the risks, as the direction seems straightforward, 
    unlike in RL problems. We will stop at this point, as all of the above suggests Adam aligns well with this description. 
    However, normally there is still much more to the "training algorithm" analysis it seems. 
=#

#= Task 7 (10 p): Plot the validation error and training error curve, where
the x-axis indicates the training epoch and the y-axis indicates the error.
=#



function main()
    # Task 1
    fct_dataset = CSV.read("covtype.csv", DataFrame)  # Forest Covertype Dataset

    println("===== Original data =====")
    inspect_data(fct_dataset)

    # Task 2
    inspect_class_distribution(fct_dataset)

    # Task 5
    train_set, val_set, test_set = split_dataset(fct_dataset)

    # Task 3
    mean_vec = Vector{Float64}()
    std_dev_vec = Vector{Float64}()

    #= Only the first 10 columns from the dataset need to be standardized, as these
    represent continues values. The rest are one-hot encoded categorical values,
    and our Cover Type class variable. 
    =#
    for i in 1:10
        col = train_set[:,i]
        mean, std_dev = get_statistics(col)
        std_col = standardize(col, mean, std_dev)
        train_set[!, i] = std_col
        push!(mean_vec, mean)
        push!(std_dev_vec, std_dev)
    end
    #= Use training set statistics to standardize validation and test sets.
    =#
    for i in 1:10
        std_col = standardize(val_set[:,i], mean_vec[i], std_dev_vec[i])
        val_set[!, i] = std_col
    end
    for i in 1:10
        std_col = standardize(test_set[:,i], mean_vec[i], std_dev_vec[i])
        test_set[!, i] = std_col
    end

    println("===== Standardized data: training set =====")
    inspect_data(train_set)
    println("===== Standardized data: validation set =====")
    inspect_data(val_set)
    println("===== Standardized data: test set =====")
    inspect_data(test_set)

    # Task 6

end

main()




