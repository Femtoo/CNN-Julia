include("helpers.jl")
# gradient
function accumulate_gradient(grad, net, d)
    dA_prev = grad
    depth=length(net["Layers"])

    dW=[]
    dB=[]

    for n in 1:depth
        layer = net["Layers"][depth-n+1]
        n_curr = depth-(n-1);
        n_prev = depth-n;
        W = nothing;
        B = nothing;
        act_fn = nothing;
        
        dA = dA_prev;
        Z = net["Z"][n_curr];
        A_prev = net["A"][n_curr];

        if typeof(layer) != MaxPoolingLayer && typeof(layer) != FlattenLayer
            W = net["Layers"][n_curr].weight;
            B = net["Layers"][n_curr].bias;
            act_fn = net["Layers"][n_curr].act_fn;
        end

        out = calculate_gradient(layer, dA, W, B, Z, A_prev, act_fn)
        
        dA_prev = out[1]
        append!(dW, [out[2]])
        append!(dB, [out[3]]) 
    end

    dW=reverse(dW)
    dB=reverse(dB)

    if d == -1
        d=[dW, dB]
    else
        for n in 1:depth
            if typeof(net["Layers"][n]) != MaxPoolingLayer && typeof(net["Layers"][n]) != FlattenLayer

                # if length(size(d[1][n])) == 4
                #     for i in 1:size(d[1][n], 1)
                #         for j in 1:size(d[1][n], 2)
                #             for k in 1:size(d[1][n], 3)
                #                 for l in 1:size(d[1][n], 4)
                #                     d[1][n][i, j, k, l] = d[1][n][i, j, k, l] + dW[n][i, j, k, l]
                #                 end
                #             end
                #         end
                #     end
                # else
                #     for i in 1:size(d[1][n], 1)
                #         for j in 1:size(d[1][n], 2)
                #             d[1][n][i, j] = d[1][n][i, j] + dW[n][i, j]
                #         end
                #     end
                # end

                # if length(size(d[2][n])) == 2
                #     for i in 1:size(d[2][n], 1)
                #         for j in 1:size(d[2][n], 2)
                #             d[2][n][i, j] = d[2][n][i, j] + dB[n][i, j]
                #         end
                #     end
                # else
                #     for i in 1:size(d[2][n], 1)
                #         d[2][n][i] = d[2][n][i] + dB[n][i]
                #     end
                # end

                d[1][n]=d[1][n].+dW[n]
                d[2][n]=d[2][n].+dB[n]
            end
        end
    end

    return d

end;

function calculate_gradient(layer::Dense, dA, W, B, Z, A_prev, act_fn)
    # d_act = d_activate(Z, act_fn)
    # println("dA: ", size(dA))
    # for i in 1:size(dA, 1)
    #     for j in 1:size(dA, 2)
    #         dZ[i, j] = dA[i, j] * tmp
    #     end
    # end
    dZ = dA.*d_activate(Z, act_fn)
    dW = (dZ * A_prev')
    dB = dZ
    dA_prev = W'*dZ
    out=[dA_prev, dW, dB]
    return out
end

function calculate_gradient(layer::FlattenLayer, dA, W, B, Z, A_prev, act_fn)
    dA = reshape(dA, layer.input_shape)
    out=[dA, nothing, nothing]
    return out
end

function calculate_gradient(layer::MaxPoolingLayer, dA::Array{Float64, 3}, W::Nothing, B::Nothing, Z::Nothing, A_prev::Array{Float64, 3}, act_fn::Nothing)
    pool_size = layer.pool_size
    stride = layer.stride
    pad = layer.pad
    # println(size(A_prev))
    # if layer.dinput === nothing
    #     layer.dinput = zeros(size(A_prev))
    # else 
    #     for i in 1:size(A_prev, 1)
    #         for j in 1:size(A_prev, 2)
    #             for c in 1:size(A_prev, 3)
    #                 layer.dinput[i, j, c] = 0
    #             end
    #         end
    #     end
    # end
    
    dinput = zeros(size(A_prev))
    # dinput = layer.dinput
    # dinput = get_buffer(layer, "dinput", size(A_prev))
    
    for c in 1:size(A_prev, 3)
        for i in 1:stride:size(A_prev, 1) - pool_size + 1
            for j in 1:stride:size(A_prev, 2) - pool_size + 1
                window = A_prev[i:i + pool_size - 1, j:j + pool_size - 1, c]
                
                max_val = maximum(window)
                
                mask = window .== max_val
                
                dinput[i:i + pool_size - 1, j:j + pool_size - 1, c] .+= dA[(i-1)÷stride + 1, (j-1)÷stride + 1, c] .* mask
            end
        end
    end
    
    out=[dinput, nothing, nothing]
    return out
end

function calculate_gradient( layer::ConvLayer, dA::Array{Float64, 3}, W::Array{Float64, 4}, B::Array{Float64, 1}, Z::Array{Float64, 3}, A_prev::Array{Float64, 3}, act_fn)
    height, width, channels = size(A_prev)
    filter_height, filter_width, _, num_filters = size(W)

    stride, padding = layer.stride, layer.pad

    grad_input = zeros(size(A_prev))
    grad_weights = zeros(size(layer.weight))
    grad_biases = zeros(size(layer.bias))

    # grad_input = get_buffer(layer, "grad_input", size(A_prev))
    # grad_weights = get_buffer(layer, "grad_weights", size(layer.weight))
    # grad_biases = get_buffer(layer, "grad_biases", size(layer.bias))

    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padded_input = zeros(Float32, padded_height, padded_width, channels)
    # padded_input = get_buffer(layer, "padded_input", (padded_height, padded_width, channels))

    padded_input[padding+1:padding+height, padding+1:padding+width, :] = A_prev
    padded_grad_input = zeros(Float32, size(padded_input))
    # padded_grad_input = get_buffer(layer, "padded_grad_input", size(padded_input))

    for k in 1:num_filters
        for h in 1:stride:padded_height-filter_height+1
            for w in 1:stride:padded_width-filter_width+1
                h_out = div(h - 1, stride) + 1
                w_out = div(w - 1, stride) + 1

                @views patch = padded_input[h:h+filter_height-1, w:w+filter_width-1, :]

                grad_bias = dA[h_out, w_out, k]
                grad_biases[k] += grad_bias
                grad_weights[:, :, :, k] += patch * grad_bias
                padded_grad_input[h:h+filter_height-1, w:w+filter_width-1, :] += layer.weight[:, :, :, k] * grad_bias
            end
        end
    end

    grad_input .= padded_grad_input[padding+1:end-padding, padding+1:end-padding, :]

    out=[grad_input, grad_weights, grad_biases]
    return out
end

# update
function update(gradients, learning_rate, net)
    for i in 1:length(net["Layers"])
        layer = net["Layers"][i]
        if !isa(layer, MaxPoolingLayer) && !isa(layer, FlattenLayer)
        W = net["Layers"][i].weight-=(learning_rate)*gradients[1][i];
        B = net["Layers"][i].bias-=(learning_rate)*gradients[2][i];
        end
    end
end;