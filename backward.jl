include("helpers.jl")
# gradient
function accumulate_gradient(grad, net)
    depth = length(net["Layers"])
    for i in 1:depth
        layer = net["Layers"][depth-i+1]
        grad = calculate_gradient(layer, grad)
        # println(typeof(layer), " grad: ", size(grad))
    end
    return grad
end;

function calculate_gradient(layer::Dense, grad)
    # d_act = d_activate(Z, act_fn)
    # println("dA: ", size(dA))
    # for i in 1:size(dA, 1)
    #     for j in 1:size(dA, 2)
    #         dZ[i, j] = dA[i, j] * tmp
    #     end
    # end
    # println(typeof(layer.act_fn))
    # println(grad)
    dZ = grad .* d_activate(layer.activation_val, layer.act_fn)

    dW = (dZ * layer.input')
    dB = sum(dZ, dims=2)
    dinput = layer.weight'*dZ
    layer.grad_weight .+= dW
    layer.grad_bias .+= dB

    return dinput
end

function calculate_gradient(layer::FlattenLayer, grad)
    grad = reshape(grad, layer.input_shape)
    return grad
end

function calculate_gradient(layer::MaxPoolingLayer, grad)
    pool_size = layer.pool_size
    stride = layer.stride
    for i in 1:layer.output_height
        for j in 1:layer.output_width
            for c in 1:layer.channels
                layer.dinput[i, j, c] = 0
            end
        end
    end
    
    for c in 1:layer.channels
        for i in 1:stride:layer.output_height - pool_size + 1
            for j in 1:stride:layer.output_width - pool_size + 1
                window = layer.output[i:i + pool_size - 1, j:j + pool_size - 1, c]
                
                max_val = maximum(window)
                
                mask = window .== max_val
                
                layer.dinput[i:i + pool_size - 1, j:j + pool_size - 1, c] .+= grad[(i-1)÷stride + 1, (j-1)÷stride + 1, c] .* mask
            end
        end
    end
    
    return layer.dinput
end

function calculate_gradient(layer::ConvLayer, grad::Array{Float64, 3})
    input = layer.pre_input

    for i in 1:layer.height
        for j in 1:layer.width
            for k in 1:layer.channels
                layer.grad_input[i, j, k] = 0.0
            end
        end
    end

    if layer.pad > 0
        layer.padded_input .= 0
        layer.padded_input[layer.pad+1:layer.pad+layer.height, layer.pad+1:layer.pad+layer.width, :] = input
    else
        layer.padded_input = input
    end

    height_out = size(grad, 1)
    width_out = size(grad, 2)

    for k in 1:layer.num_filters
        for h in 1:layer.stride:(layer.height - layer.filter_height + 1 + 2 * layer.pad)
            for w in 1:layer.stride:(layer.width - layer.filter_width + 1 + 2 * layer.pad)
                h_out = div(h - 1, layer.stride) + 1
                w_out = div(w - 1, layer.stride) + 1
                if h_out <= height_out && w_out <= width_out
                    @views patch = layer.padded_input[h:h+layer.filter_height-1, w:w+layer.filter_width-1, :]
                    grad_bias = grad[h_out, w_out, k]
                    layer.grad_bias[k] += grad_bias
                    for kh in 1:layer.filter_height
                        for kw in 1:layer.filter_width
                            for kc in 1:layer.channels
                                layer.grad_weight[kh, kw, kc, k] += patch[kh, kw, kc] * grad_bias
                            end
                        end
                    end
                    layer.padded_grad_input[h:h+layer.filter_height-1, w:w+layer.filter_width-1, :] = layer.weight[:, :, :, k] * grad_bias
                end
            end
        end
    end

    layer.grad_input .= layer.padded_grad_input[layer.pad+1:end-layer.pad, layer.pad+1:end-layer.pad, :]

    return layer.grad_input
end

# update
function update(learning_rate, net)
    for i in 1:length(net["Layers"])
        layer = net["Layers"][i]
        if !isa(layer, MaxPoolingLayer) && !isa(layer, FlattenLayer)
            # println("name: ", typeof(layer), layer.grad_weight)
            layer.grad_weight ./= 100
            layer.grad_bias ./= 100

            layer.weight .-= (learning_rate) * layer.grad_weight;
            layer.bias .-= (learning_rate) * layer.grad_bias;
            # exit()
        end
    end
end;