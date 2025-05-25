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




