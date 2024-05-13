# forward
function forward(x::AbstractArray, net::Dict)
    A=Any[x]
    Z=Any[]
    for n in 1:length(net["Layers"])
        # layer = net["Layers"][n]
        x, z = forward_layer(x, net["Layers"][n])
        push!(Z, z)
        push!(A, x)
    end

    net["A"]=A
    net["Z"]=Z
    return x
end; 

function forward_layer(x::AbstractArray, layer::ConvLayer)
    z = convolve(x, layer)
    x = activate(z, layer.act_fn)
    return x, z
end

function forward_layer(x::AbstractArray, layer::Dense)
    z = layer.weight*x + layer.bias
    x = activate(z, layer.act_fn)
    return x, z
end

function forward_layer(x::AbstractArray, layer::MaxPoolingLayer)
    output_height = div(size(x, 1) - layer.pool_size + 2 * layer.pad, layer.stride) + 1
    output_width = div(size(x, 2) - layer.pool_size + 2 * layer.pad, layer.stride) + 1
    output_channels = size(x, 3)
    output = zeros(output_height, output_width, output_channels)
    # output = get_buffer(layer, "output", (output_height, output_width, output_channels))
    # println(size(output))

    # z = copy(x)

    for c in 1:output_channels
        for i in 1:layer.stride:(size(x, 1) - layer.pool_size + 1)
            for j in 1:layer.stride:(size(x, 2) - layer.pool_size + 1)
                @views window = x[i:(i + layer.pool_size - 1), j:(j + layer.pool_size - 1), c]
                output[div(i-1, layer.stride) + 1, div(j-1, layer.stride) + 1, c] = maximum(window)
            end
        end
    end
    # return output, z
    return output, nothing
end
function forward_layer(x::AbstractArray, layer::FlattenLayer)
    layer.input_shape = size(x)
    x = reshape(x, 400, 1)
    return x, nothing
end

function convolve(input::Array{Float64, 3}, layer::ConvLayer)::Array{Float64, 3}
    height, width, channels = size(input)
    filter_height, filter_width, _, num_filters = size(layer.weight)
    
    output_height = div(height - filter_height + 2 * layer.pad, layer.stride) + 1
    output_width = div(width - filter_width + 2 * layer.pad, layer.stride) + 1

    output = zeros(output_height, output_width, num_filters)
    # output = get_buffer(layer, "outputf", (output_height, output_width, num_filters))

    padded_input = zeros(Float32, height + 2 * layer.pad, width + 2 * layer.pad, channels)
    # padded_input = get_buffer(layer, "padded_inputf", (height + 2 * layer.pad, width + 2 * layer.pad, channels))
    padded_input[layer.pad+1:layer.pad+height, layer.pad+1:layer.pad+width, :] = input

    for k in 1:num_filters
        filter = layer.weight[:, :, :, k]
        for h in 1:layer.stride:height-filter_height+1+2*layer.pad
            for w in 1:layer.stride:width-filter_width+1+2*layer.pad
                @views patch = padded_input[h:h+filter_height-1, w:w+filter_width-1, :]
                sum_val = 0.0

                for kh in 1:filter_height
                    for kw in 1:filter_width
                        for kc in 1:channels
                            sum_val += patch[kh, kw, kc] * filter[kh, kw, kc]
                        end
                    end
                end
                # output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, k] += sum(patch .* filter)

                output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, k] += sum_val
            end
        end
    end

    return output
end