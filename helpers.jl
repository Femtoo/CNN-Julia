function get_buffer(layer, key, dims...)
    # println("layer: ", typeof(layer), " in ", key)
    # if layer.buffer[key] === nothing || any(size(layer.buffer[key]) .!= dims)
    if layer.buffer[key] === nothing
        # println("made new")
        layer.buffer[key] = zeros(dims...)
    else
        # println("filled")
        fill!(layer.buffer[key], 0)
    end
    return layer.buffer[key]
end


function zi_one_hot_encode(data)
    one_hot=zeros(Float64, maximum(data)+1, size(data, 1));
    for i in 1:size(data, 1)
        label=data[i]+1
        one_hot[label, i]=1
    end;
    return one_hot
end;