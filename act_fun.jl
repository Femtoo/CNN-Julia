using Statistics: mean

#funkcje aktywacji
struct ReLU
end;
activate(x, act_fn::ReLU) = max.(0, x)
d_activate(x, act_fn::ReLU) = 1.0 .* (x .> 0)

struct Softmax
end;
activate(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = softmax(x)
d_activate(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = (softmax(x)) .* (1 .- softmax(x))

struct Identity
end;
activate(x::Array{Float64, 2}, act_fn::Identity)::Array{Float64,2} = x
d_activate(x::Array{Float64, 2}, act_fn::Identity)::Array{Float64,2} = ones(size(x))

function softmax(x)
    c=maximum(x)
    p = x .- log.( sum( exp.(x .- c) ) ) .-c
    p = exp.(p)
    return p
end

function one_cold(encoded)
    return [argmax(vec) for vec in eachcol(encoded)]
end

function cross_entropy_loss_with_gradient(predictions, targets)
    probabilities = softmax(predictions)
    loss = -mean(sum(targets .* log.(probabilities), dims=1))
    gradient = probabilities - targets
    return loss, gradient
end

function loss_and_accuracy(ŷ, y)
    loss, grad = cross_entropy_loss_with_gradient(ŷ, y)
    pred_classes = one_cold(ŷ)
    true_classes = one_cold(y)
    acc = round(100 * mean(pred_classes .== true_classes); digits=2)
    return loss, acc, grad
end

#funkcje straty
# xe_loss(y_hat, y) = -sum(y.*log.(softmax(y_hat)))
function xe_loss(y_hat, y) 
    probs = softmax(y_hat)
    return -sum(y.*log.(probs))
end
xe_loss_derivative(y_hat, y) = y_hat - y