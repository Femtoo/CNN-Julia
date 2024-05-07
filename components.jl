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

#funkcje straty
# xe_loss(y_hat, y) = -sum(y.*log.(softmax(y_hat)))
xe_loss(y_hat, y) = -sum(y.*log.(y_hat))
xe_loss_derivative(y_hat, y) = y_hat - y


# warstwy
mutable struct Dense
    weight
    bias
    act_fn
    Dense(dim_in, dim_out, act_fn)=new(kaiming(Float64, dim_out, dim_in), zeros(Float64, dim_out, 1), act_fn)
end;

mutable struct ConvLayer
    weight::Array{Float64, 4}
    bias::Array{Float64, 1}
    stride::Int
    pad::Int
    act_fn

    function ConvLayer(filter_height::Int, filter_width::Int, in_channels::Int, out_channels::Int; stride::Int=1, pad::Int=0, act_fn)
        new(randn(Float64, filter_height, filter_width, in_channels, out_channels) .* sqrt(2 / (filter_height * filter_width * in_channels)), 
            zeros(Float64, out_channels), 
            stride, 
            pad,
            act_fn)
    end
end

mutable struct MaxPoolingLayer
    pool_size::Int
    stride::Int
    pad::Int

    function MaxPoolingLayer(pool_size::Int; stride::Int, pad::Int=0)
        new(pool_size, stride, pad)
    end
end

mutable struct FlattenLayer
    input_shape::Union{Nothing, Tuple{Int, Int, Int}}
    FlattenLayer() = new(nothing)
end

function kaiming(type, dim_out,dim_in)
    matrix = randn(type, dim_out, dim_in).*sqrt(2/dim_in)
    return matrix
end