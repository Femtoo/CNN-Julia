using LinearAlgebra
using MLDatasets
using Plots

include("helpers.jl");
include("components.jl");
include("act_fun.jl");
include("train.jl");


#network
net=Dict("Layers"=>[], "A"=>Any[], "Z"=>Any[])

layers=[]
append!(layers, [ConvLayer(3, 3, 1, 6, act_fn = ReLU())])
append!(layers, [MaxPoolingLayer(2, stride=2, pad=0)])
append!(layers, [ConvLayer(3, 3, 6, 16, act_fn = ReLU())])
append!(layers, [MaxPoolingLayer(2, stride=2, pad=0)])
append!(layers, [FlattenLayer()])
append!(layers, [Dense(400, 84, ReLU())])
append!(layers, [Dense(84, 10, Softmax())])
net["Layers"]=layers;

#init the training data
train_x, train_y = MNIST.traindata(Float64);
test_x, test_y = MNIST.testdata(Float64);

train_y_one_hot=zi_one_hot_encode(train_y);
test_y_one_hot=zi_one_hot_encode(test_y);


#train
mb_size=100
epochs=1
lr=1e-2

results=train(net, mb_size, lr, epochs, train_x, test_x, train_y_one_hot, test_y_one_hot);


# #plot
# plot([results["training_loss"], results["test_loss"]], lw=2, label=["Train" "Test"], legend=:topright)
# xlabel!("Epoch")
# ylabel!("Loss")
# savefig("training.png")

# plot([results["training_acc"], results["test_acc"]], lw=2, label=["Train" "Test"], legend=:bottomright)
# xlabel!("Epoch")
# ylabel!("Accuracy")
# savefig("accuracy.png")