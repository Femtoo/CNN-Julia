include("backward.jl")
include("forward.jl")
include("act_fun.jl")

function train(net, batch_size, lr, epochs, train_x, test_x, train_y_one_hot, test_y_one_hot)
    batch_acc = []
    loss_results = []

    for epoch in 1:epochs

        epoch_train_loss=[]
        epoch_train_correct_count=0.0
        batch_train_correct_count=0.0

        batch_counter=1;
        batch_loss=0.0;
        
        println("Running epoch ", epoch)
        
        @time begin
            for i in 1:size(train_y,1)
                
                x_tr = reshape(train_x[:, :, i], 28, 28, 1)
                y_hat_tr =forward(x_tr, net);
                y_tr = reshape(train_y_one_hot[:, i], 10, 1);

                loss, grad_loss = cross_entropy_loss(y_hat_tr, y_tr)

                accumulate_gradient(grad_loss, net)

                epoch_train_correct_count+=(argmax(y_hat_tr) == argmax(y_tr))
                batch_train_correct_count+=(argmax(y_hat_tr) == argmax(y_tr))

                batch_loss+=loss
                
                if batch_counter%batch_size==0

                    update(lr, net, batch_size)
                    
                    append!(epoch_train_loss, (batch_loss / batch_size))
                    append!(loss_results, (batch_loss / batch_size))
                    append!(batch_acc, batch_train_correct_count/batch_size)
                    batch_counter=0;
                    batch_loss=0.0;
                    batch_train_correct_count=0.0
                end

                batch_counter+=1
            end
        end

        println("Train Accuracy: ", epoch_train_correct_count/size(train_y,1))
        println("Train Loss: ", sum(epoch_train_loss)/length(epoch_train_loss))
    end

    test_correct_count=0.0
    test_loss=[]

    for i in 1:size(test_y,1)

        x_te=test_x[:,:,i];
        x_te = reshape(test_x[:, :, i], 28, 28, 1)
        y_hat_te=forward(x_te, net);
        y_te=reshape(test_y_one_hot[:,i], 10, 1);
        loss, grad_loss = cross_entropy_loss(y_hat_te, y_te)
        append!(test_loss, loss)

        test_correct_count+=(argmax(y_hat_te)==argmax(y_te))

    end

    println("Test Accuracy: ", test_correct_count/size(test_y,1))
    println("Test Loss: ", sum(test_loss)/length(test_loss))

    return batch_acc, loss_results
end