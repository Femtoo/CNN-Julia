# forward
function forward(x::AbstractArray, net::Dict)

    for n in 1:length(net["Layers"])
        # println(typeof(net["Layers"][n]), " input: ", size(x))
        x = forward_layer(x, net["Layers"][n])
    end

    return x
end; 

function forward_layer(x::AbstractArray, layer::ConvLayer)
    layer.pre_input = x
    z = convolve(x, layer)
    for c in axes(z, 3)
        z[:, :, c] .+= layer.bias[c]
    end
    x = activate(z, layer.act_fn)
    return x
end

function forward_layer(x::AbstractArray, layer::Dense)
    layer.input = x
    z = layer.weight*x + layer.bias
    x = activate(z, layer.act_fn)
    layer.activation_val = x
    return layer.activation_val
end

function forward_layer(x::AbstractArray, layer::MaxPoolingLayer)
    # output_height = div(size(x, 1) - layer.pool_size + 2 * layer.pad, layer.stride) + 1
    # output_width = div(size(x, 2) - layer.pool_size + 2 * layer.pad, layer.stride) + 1
    # output_channels = size(x, 3)
    # output = zeros(output_height, output_width, output_channels)
    # output = get_buffer(layer, "output", (output_height, output_width, output_channels))
    # println(size(output))

    # z = copy(x)
    
    # UWAGA TUTAJ MOZNA JESZCZE ZEROWAC OUTPUT PRZED WYKONANIEM???!!!

    for c in 1:layer.channels
        for i in 1:layer.stride:(layer.input_height - layer.pool_size + 1)
            for j in 1:layer.stride:(layer.input_width - layer.pool_size + 1)
                @views window = x[i:(i + layer.pool_size - 1), j:(j + layer.pool_size - 1), c]
                layer.output[div(i-1, layer.stride) + 1, div(j-1, layer.stride) + 1, c] = maximum(window)
            end
        end
    end
    
    return layer.output
end
function forward_layer(x::AbstractArray, layer::FlattenLayer)
    layer.input_shape = size(x)
    x = reshape(x, 400, 1)
    return x
end

function convolve(input::Array{Float64, 3}, layer::ConvLayer)::Array{Float64, 3}
    if layer.pad > 0
        for i in 1:(layer.height+2*layer.pad)
            for j in 1:(layer.width+2*layer.pad)
                for k in 1:layer.channels
                    layer.padded_input[i, j, k] = 0.0
                end
            end
        end
        layer.padded_input[layer.pad+1:layer.pad+layer.height, layer.pad+1:layer.pad+layer.width, :] = input
    else
        layer.padded_input = input
    end

    for k in 1:size(layer.output, 3)
        for i in 1:size(layer.output, 1)
            for j in 1:size(layer.output, 2)
                layer.output[i, j, k] = 0.0
            end
        end
    end
    
    for f in 1:layer.num_filters
        filter = layer.weight[:, :, :, f]
        for h in 1:layer.stride:layer.height-layer.filter_height+1+2*layer.pad
            for w in 1:layer.stride:layer.width-layer.filter_width+1+2*layer.pad
                @views patch = layer.padded_input[h:h+layer.filter_height-1, w:w+layer.filter_width-1, :]
                sum_val = 0.0

                for fh in 1:layer.filter_height
                    for fw in 1:layer.filter_width
                        for fc in 1:layer.channels
                            sum_val += patch[fh, fw, fc] * filter[fh, fw, fc]
                        end
                    end
                end
                # output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, k] += sum(patch .* filter)

                layer.output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, f] += sum_val
            end
        end
    end

    return layer.output
end

# function convolve(input::Array{Float64, 3}, layer::ConvLayer)::Array{Float64, 3}
#     height, width, channels = size(input)
#     filter_height, filter_width, _, num_filters = size(layer.weight)
    
#     output_height = div(height - filter_height + 2 * layer.pad, layer.stride) + 1
#     output_width = div(width - filter_width + 2 * layer.pad, layer.stride) + 1

#     output = zeros(output_height, output_width, num_filters)
#     # output = get_buffer(layer, "outputf", (output_height, output_width, num_filters))

#     padded_input = zeros(Float32, height + 2 * layer.pad, width + 2 * layer.pad, channels)
#     # padded_input = get_buffer(layer, "padded_inputf", (height + 2 * layer.pad, width + 2 * layer.pad, channels))
#     padded_input[layer.pad+1:layer.pad+height, layer.pad+1:layer.pad+width, :] = input

#     for k in 1:num_filters
#         filter = layer.weight[:, :, :, k]
#         for h in 1:layer.stride:height-filter_height+1+2*layer.pad
#             for w in 1:layer.stride:width-filter_width+1+2*layer.pad
#                 @views patch = padded_input[h:h+filter_height-1, w:w+filter_width-1, :]
#                 sum_val = 0.0

#                 for kh in 1:filter_height
#                     for kw in 1:filter_width
#                         for kc in 1:channels
#                             sum_val += patch[kh, kw, kc] * filter[kh, kw, kc]
#                         end
#                     end
#                 end
#                 # output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, k] += sum(patch .* filter)

#                 output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, k] += sum_val
#             end
#         end
#     end

#     return output
# end