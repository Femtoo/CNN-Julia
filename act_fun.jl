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

function cross_entropy_loss(y_hat, y)
    probabilities = softmax(y_hat)
    loss = -mean(sum(y .* log.(probabilities), dims=1))
    gradient = probabilities - y
    return loss, gradient
end