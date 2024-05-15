# warstwy
mutable struct Dense
    weight::Array{Float64,2}
    bias::Array{Float64,1}
    grad_weight::Array{Float64,2}
    grad_bias::Array{Float64,1}
    act_fn
    activation_val::Array{Float64,2}
    input::Array{Float64,2}
    function Dense(dim_in, dim_out, act_fn)
        new(kaiming(Float64, dim_out, dim_in),
        zeros(Float64, dim_out),
        zeros(Float64, dim_out, dim_in),
        zeros(Float64, dim_out),
        act_fn,
        zeros(Float64, dim_out, 1),
        zeros(Float64, dim_in, 1)
        )
    end
end;

mutable struct ConvLayer
    weight::Array{Float64, 4}
    bias::Array{Float64, 1}
    grad_weight::Array{Float64, 4}
    grad_bias::Array{Float64, 1}
    stride::Int
    pad::Int
    act_fn
    pre_input::Array{Float64, 3}
    height::Int
    width::Int
    channels::Int
    filter_height::Int
    filter_width::Int
    num_filters::Int
    output_height::Int
    output_width::Int
    output::Array{Float64, 3}
    padded_input::Array{Float64, 3}
    grad_input::Array{Float64, 3}
    padded_grad_input::Array{Float64, 3}

    function ConvLayer(filter_height::Int, filter_width::Int, in_channels::Int, out_channels::Int; stride::Int=1, pad::Int=0, act_fn, input_height::Int, input_width::Int, input_channels::Int)
        output_height = div(input_height - filter_height + 2 * pad, stride) + 1
        output_width = div(input_width - filter_width + 2 * pad, stride) + 1

        new(kaiming_conv(Float64, filter_height, filter_width, in_channels, out_channels),
            zeros(Float64, out_channels), 
            zeros(Float64, filter_height, filter_width, in_channels, out_channels),
            zeros(Float64, out_channels),
            stride, 
            pad,
            act_fn,
            zeros(Float64, input_height, input_width, in_channels),
            input_height,
            input_width,
            in_channels,
            filter_height,
            filter_width,
            out_channels,
            output_height,
            output_width,
            zeros(Float64, output_height, output_width, out_channels),
            zeros(Float64, input_height + 2 * pad, input_width + 2 * pad, in_channels),
            zeros(Float64, input_height, input_width, in_channels),
            zeros(Float64, input_height + 2 * pad, input_width + 2 * pad, in_channels)
            )
    end
end

mutable struct MaxPoolingLayer
    pool_size::Int
    stride::Int
    pad::Int
    input_height::Int
    input_width::Int
    channels::Int
    output_height::Int
    output_width::Int
    dinput::Union{Nothing, Array{Float64, 3}}
    output::Array{Float64, 3}

    function MaxPoolingLayer(pool_size::Int; stride::Int, pad::Int, input_height::Int, input_width::Int, input_channels::Int)
        output_height = div(input_height - pool_size + 2 * pad, stride) + 1
        output_width = div(input_width - pool_size + 2 * pad, stride) + 1

        new(pool_size, 
        stride, 
        pad, 
        input_height,
        input_width,
        input_channels,
        output_height,
        output_width,
        zeros(output_height, output_width, input_channels), 
        zeros(output_height, output_width, input_channels)
        )
    end
end

mutable struct FlattenLayer
    input_shape::Union{Nothing, Tuple{Int, Int, Int}}
    FlattenLayer() = new(nothing)
end

function kaiming(type, dim_out,dim_in)
    multiplier = sqrt(2/dim_in)
    matrix = randn(type, dim_out, dim_in)
    for i in 1:dim_out
        for j in 1:dim_in
            matrix[i, j] *= multiplier
        end
    end
    return matrix
end

function kaiming_conv(type, filter_height, filter_width, in_channels, out_channels)
    multiplier = sqrt(2 / (filter_height * filter_width * in_channels))
    matrix = randn(type, filter_height, filter_width, in_channels, out_channels)
    for i in 1: filter_height
        for j in 1: filter_width
            for k in 1: in_channels
                for l in 1: out_channels
                    matrix[i, j, k, l] *= multiplier
                end
            end
        end
    end
    return matrix
end
