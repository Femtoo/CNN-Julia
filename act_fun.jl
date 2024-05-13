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
d_activate(x::Array{Float64, 2}, act_fn::Identity)::Array{Float64,2} = fill!(x, 0)

function softmax(x)
    c=maximum(x)
    p = x .- log.( sum( exp.(x .- c) ) ) .-c
    p = exp.(p)
    return p
end

#funkcje straty
# xe_loss(y_hat, y) = -sum(y.*log.(softmax(y_hat)))
xe_loss(y_hat, y) = -sum(y.*log.(y_hat))
xe_loss_derivative(y_hat, y) = y_hat - y