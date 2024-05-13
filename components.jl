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
    buffer::Dict{String, Any}

    function ConvLayer(filter_height::Int, filter_width::Int, in_channels::Int, out_channels::Int; stride::Int=1, pad::Int=0, act_fn)
        new(randn(Float64, filter_height, filter_width, in_channels, out_channels) .* sqrt(2 / (filter_height * filter_width * in_channels)), 
            zeros(Float64, out_channels), 
            stride, 
            pad,
            act_fn,
            Dict("grad_input" => nothing, "grad_weights" => nothing, "grad_biases" => nothing, "padded_input" => nothing, "padded_grad_input" => nothing, "outputf" => nothing, "padded_inputf" => nothing))
    end
end

mutable struct MaxPoolingLayer
    pool_size::Int
    stride::Int
    pad::Int
    buffer::Dict{String, Any}

    function MaxPoolingLayer(pool_size::Int; stride::Int, pad::Int=0)
        new(pool_size, stride, pad, Dict("dinput" => nothing, "output" => nothing))
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