# forward
function forward(x::AbstractArray, net::Dict)
    A=Any[x]
    Z=Any[]
    for n in 1:length(net["Layers"])
        layer = net["Layers"][n]
        x, z = forward_layer(x, layer)
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

    z = copy(x)

    for c in 1:output_channels
        for i in 1:layer.stride:(size(x, 1) - layer.pool_size + 1)
            for j in 1:layer.stride:(size(x, 2) - layer.pool_size + 1)
                window = x[i:(i + layer.pool_size - 1), j:(j + layer.pool_size - 1), c]
                output[div(i-1, layer.stride) + 1, div(j-1, layer.stride) + 1, c] = maximum(window)
            end
        end
    end
    return output, z
end
function forward_layer(x::AbstractArray, layer::FlattenLayer)
    layer.input_shape = size(x)
    x = reshape(x, 400, 1)
    return x, x
end

function convolve(input::Array{Float64, 3}, layer::ConvLayer)::Array{Float64, 3}
    height, width, channels = size(input)
    filter_height, filter_width, _, num_filters = size(layer.weight)
    
    output_height = div(height - filter_height + 2 * layer.pad, layer.stride) + 1
    output_width = div(width - filter_width + 2 * layer.pad, layer.stride) + 1

    output = zeros(output_height, output_width, num_filters)

    padded_input = zeros(Float32, height + 2 * layer.pad, width + 2 * layer.pad, channels)
    padded_input[layer.pad+1:layer.pad+height, layer.pad+1:layer.pad+width, :] = input

    for k in 1:num_filters
        filter = layer.weight[:, :, :, k]
        for h in 1:layer.stride:height-filter_height+1+2*layer.pad
            for w in 1:layer.stride:width-filter_width+1+2*layer.pad
                patch = padded_input[h:h+filter_height-1, w:w+filter_width-1, :]
                output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, k] += sum(patch .* filter)
            end
        end
    end

    return output
end

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
                d[1][n]=d[1][n].+dW[n]
                d[2][n]=d[2][n].+dB[n]
            end
        end
    end

    return d

end;

function calculate_gradient(layer::Dense, dA, W, B, Z, A_prev, act_fn)
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

function calculate_gradient(layer::MaxPoolingLayer, dA::Array{Float64, 3}, W::Nothing, B::Nothing, Z::Array{Float64, 3}, A_prev::Array{Float64, 3}, act_fn::Nothing)
    pool_size = layer.pool_size
    stride = layer.stride
    pad = layer.pad
    
    dinput = zeros(size(A_prev))
    
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

    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padded_input = zeros(Float32, padded_height, padded_width, channels)
    padded_input[padding+1:padding+height, padding+1:padding+width, :] = A_prev
    padded_grad_input = zeros(Float32, size(padded_input))

    for k in 1:num_filters
        for h in 1:stride:padded_height-filter_height+1
            for w in 1:stride:padded_width-filter_width+1
                h_out = div(h - 1, stride) + 1
                w_out = div(w - 1, stride) + 1

                patch = padded_input[h:h+filter_height-1, w:w+filter_width-1, :]

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