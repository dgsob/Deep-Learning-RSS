# Tutorial: Building a CNN with Lux.jl

# Import packages
using Lux, MLDatasets, Optimisers, Random, Statistics, Zygote, JLD2
using Lux: Chain, Conv, Dense, MaxPool, FlattenLayer, Dropout
using LuxCore
using MLDatasets: CIFAR10

# Auxiliary functions 

# Softmax: exp(x)/sum(exp(x))
function softmax(y, dims=1)
    y_max = maximum(y, dims=dims)
    return exp.(y .- y_max) ./ sum(exp.(y .- y_max), dims=dims)
end

# Cross-entropy: -sum(y * log(y_pred))
function cross_entropy(y, y_pred, dims=1)
    y_pred = clamp.(y_pred, eps(Float32), 1.0)
    return -mean(sum(y .* log.(y_pred), dims=dims))
end

# One-hot encoding
function onehot(y)
    return Float32.(unique(y) .== permutedims(y))
end

# One-cold: converts from one-hot encoded labels or softmax outputs to true numerical labels
function onecold(y)
    return [argmax(y[:, i]) - 1 for i in axes(y, 2)]
end

# Step 1: Load and preprocess CIFAR-10 dataset
function load_cifar10()
    train_data = CIFAR10(:train)
    test_data = CIFAR10(:test)
    
    # Convert images to Float32 and normalize to [0,1]
    train_x = Float32.(train_data.features) ./ 255
    train_y = train_data.targets
    test_x = Float32.(test_data.features) ./ 255
    test_y = test_data.targets
    
    # Reshape: (width, height, channels, batch) -> (height, width, channels, batch)
    if ndims(train_x) == 4 && size(train_x, 3) == 3 && size(train_x, 1) == 32
        train_x = permutedims(train_x, (2, 1, 3, 4))
        test_x = permutedims(test_x, (2, 1, 3, 4))
    else
        error("Unexpected input dimensions: $(size(train_x))")
    end
    
    # One-hot encode labels (10 classes)
    train_y = onehot(train_y)
    test_y = onehot(test_y)
    
    return (train_x, train_y), (test_x, test_y)
end

# Step 2: Define the CNN model
function create_model()
    Chain(
        Conv((3, 3), 3 => 32, relu, pad=(1, 1)),  # Input: 32x32x3, Output: 32x32x32
        MaxPool((2, 2)),                          # Output: 16x16x32
        Conv((3, 3), 32 => 64, relu, pad=(1, 1)), # Output: 16x16x64
        MaxPool((2, 2)),                          # Output: 8x8x64
        Conv((3, 3), 64 => 64, relu, pad=(1, 1)), # Output: 8x8x64
        MaxPool((2, 2)),                          # Output: 4x4x64
        FlattenLayer(),                           # Output: 1024 (4*4*64)
        Dense(1024, 512, relu),
        Dropout(0.1),                             # 
        Dense(512, 10)                            # Output: 10 (logits)
    )
end

# Step 3: Define loss function with manual softmax and cross-entropy
function loss_function(model, θ, ξ, x, y)
    y_pred, ξ = model(x, θ, ξ)
    y_pred = softmax(y_pred)
    loss = cross_entropy(y, y_pred)
    return loss, ξ, y_pred
end

# Step 4: Process a single batch
function process_batch!(model, θ, ξ, x_batch, y_batch, opt_state)
    (loss, ξ, y_pred), back = Zygote.pullback(p -> loss_function(model, p, ξ, x_batch, y_batch), θ)
    grads = back((1.0, nothing, nothing))[1]
    opt_state, θ = Optimisers.update(opt_state, θ, grads)
    # Compute batch accuracy
    pred_indices = onecold(y_pred)
    true_indices = onecold(y_batch)
    accuracy = mean(pred_indices .== true_indices)
    return loss, ξ, opt_state, θ, accuracy
end

# Step 5: Training function
function train!(model, θ, ξ, train_data; epochs=10, batch_size=128)
    train_x, train_y = train_data
    
    # Setup optimizer
    opt = Optimisers.Adam(0.001)
    opt_state = Optimisers.setup(opt, θ)
    
    # Manual batching
    n_samples = size(train_x)[end]
    
    for epoch in 1:epochs
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        indices = shuffle(1:n_samples)
        
        for i in 1:batch_size:n_samples
            batch_indices = indices[i:min(i+batch_size-1, n_samples)]
            x_batch = train_x[:, :, :, batch_indices]
            y_batch = train_y[:, batch_indices]
            
            # Process batch
            loss, ξ, opt_state, θ, accuracy = process_batch!(model, θ, ξ, x_batch, y_batch, opt_state)
            
            total_loss += loss
            total_accuracy += accuracy
            n_batches += 1
        end
        
        println("Epoch $epoch, Loss: $(total_loss/n_batches), Train Accuracy: $(total_accuracy/n_batches)")
    end
    
    return θ, ξ
end

# Step 6: Evaluation function
function evaluate(model, θ, ξ, test_data)
    test_x, test_y = test_data
    ξ_test = LuxCore.testmode(ξ)  # Switch to test mode to disable dropout
    y_pred, _ = model(test_x, θ, ξ_test)
    y_pred = softmax(y_pred)
    pred_indices = onecold(y_pred)
    true_indices = onecold(test_y)
    accuracy = mean(pred_indices .== true_indices)
    println("Test Accuracy: $accuracy")
end

# Step 7: Main function to run training and evaluation
function main()
    # Load data
    (train_x, train_y), (test_x, test_y) = load_cifar10()
    
    # Verify shapes
    println("train_x size: ", size(train_x), ", train_y size: ", size(train_y))
    println("test_x size: ", size(test_x), ", test_y size: ", size(test_y))
    
    # Create and initialize model
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    model = create_model()
    θ, ξ = Lux.setup(rng, model)
    
    # Load pretrained model if exists
    model_file = "model.jld2"
    if isfile(model_file)
        try
            saved = JLD2.load(model_file)
            θ = saved["θ"]
            ξ = saved["ξ"]
            println("Loaded model from $model_file")
        catch e
            println("Failed to load model: $e. Training new model...")
            θ, ξ = train!(model, θ, ξ, (train_x, train_y))
            JLD2.save(model_file, Dict("θ" => θ, "ξ" => ξ))
            println("Saved model to $model_file")
        end
    else
        println("Model file not found. Training new model...")
        θ, ξ = train!(model, θ, ξ, (train_x, train_y))
        JLD2.save(model_file, Dict("θ" => θ, "ξ" => ξ))
        println("Saved model to $model_file")
    end
    
    # Evaluate model
    evaluate(model, θ, ξ, (test_x, test_y))
end

# Run the main function
main()