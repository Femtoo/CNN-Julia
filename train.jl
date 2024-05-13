include("backpropagation.jl")
include("forward.jl")

function train(net, batch_size, lr, epochs, train_x, test_x, train_y_one_hot, test_y_one_hot)

    average_epoch_train_loss=[]
    average_epoch_test_loss=[]
    average_epoch_train_acc=[]
    average_epoch_test_acc=[]

    @time for epoch in 1:epochs

        epoch_train_loss=[]
        epoch_train_correct_count=0.0

        batch_counter=1;
        loss=0.0;
        d=-1
        
        println("Running epoch ", epoch)

        # for i in 1:size(train_y,1)
        for i in 1:3

            if i % 10000 == 0
                println(i)
            end
            
            x_tr = reshape(train_x[:, :, i], 28, 28, 1)
            y_hat_tr = forward(x_tr, net);
            y_tr = reshape(train_y_one_hot[:, i], 10, 1);

            loss_grad = xe_loss_derivative(y_hat_tr, y_tr)./batch_size

            d = accumulate_gradient(loss_grad, net, d)

            loss = loss.+(xe_loss(y_hat_tr, y_tr)./batch_size)
            epoch_train_correct_count+=(argmax(y_hat_tr) == argmax(y_tr))
            
            if batch_counter%batch_size==0

                update(d, lr, net)
                
                append!(epoch_train_loss, loss)

                d=-1;
                batch_counter=0;
                loss=0.0;

            end

            batch_counter+=1

        end

        println("Train Accuracy: ", epoch_train_correct_count/size(train_y,1))

        # append!(average_epoch_train_loss, sum(epoch_train_loss)/length(epoch_train_loss))
        append!(average_epoch_train_acc, epoch_train_correct_count/size(train_y,1))
        
        # append!(average_epoch_test_loss, sum(epoch_test_loss)/length(epoch_test_loss))
        # append!(average_epoch_test_acc, epoch_test_correct_count/size(test_y,1))

    end

    test_correct_count=0.0
    test_loss=[]

    # for i in 1:size(test_y,1)
    for i in 1:3

        x_te=test_x[:,:,i];
        x_te = reshape(test_x[:, :, i], 28, 28, 1)
        y_hat_te=forward(x_te, net);
        y_te=reshape(test_y_one_hot[:,i], 10, 1);

        append!(test_loss, xe_loss(y_hat_te,y_te))
        test_correct_count+=(argmax(y_hat_te)==argmax(y_te))

    end

    println("Test Accuracy: ", test_correct_count/size(test_y,1))
    println("Test Loss: ", sum(test_loss)/length(test_loss))

    results = Dict("training_loss"=>average_epoch_train_loss, 
                 "test_loss"=>average_epoch_test_loss, 
                 "training_acc"=>average_epoch_train_acc,
                 "test_acc"=>average_epoch_test_acc)

    return results
end