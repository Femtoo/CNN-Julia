using LinearAlgebra
using MLDatasets
using Plots

include("helpers.jl");
include("components.jl");
include("act_fun.jl");
include("training.jl");


#network
net=Dict("Layers"=>[])

layers=[]
append!(layers, [ConvLayer(3, 3, 1, 6, act_fn = ReLU(), input_height = 28, input_width = 28, input_channels = 1)])
append!(layers, [MaxPoolingLayer(2, stride=2, pad=0, input_height=26, input_width=26, input_channels=6)])
append!(layers, [ConvLayer(3, 3, 6, 16, act_fn = ReLU(), input_height = 13, input_width = 13, input_channels = 6)])
append!(layers, [MaxPoolingLayer(2, stride=2, pad=0, input_height=11, input_width=11, input_channels=16)])
append!(layers, [FlattenLayer()])
append!(layers, [Dense(400, 84, ReLU())])
append!(layers, [Dense(84, 10, Identity())])
net["Layers"]=layers;

#init the training data
train_x, train_y = MNIST.traindata(Float64);
test_x, test_y = MNIST.testdata(Float64);

train_y_one_hot=zi_one_hot_encode(train_y);
test_y_one_hot=zi_one_hot_encode(test_y);


#train
mb_size=100
epochs=3
lr=1e-2

acc_results, loss_results=train(net, mb_size, lr, epochs, train_x, test_x, train_y_one_hot, test_y_one_hot);

plot(collect(1:length(acc_results)), acc_results, lw=2, label=["Training accuracy"], legend=:bottomright)
xlabel!("Batch")
ylabel!("Accuracy")

epoki = [600, 1200, 1800] 
xticks!(epoki, ["Epoch 1", "Epoch 2", "Epoch 3"])

savefig("accuracy_batch.png")

plot(collect(1:length(loss_results)), loss_results, lw=2, label=["Training loss"], legend=:bottomright)
xlabel!("Batch")
ylabel!("Loss")

epoki = [600, 1200, 1800] 
xticks!(epoki, ["Epoch 1", "Epoch 2", "Epoch 3"])

savefig("loss_batch.png")