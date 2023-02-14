load("../data/ct_data.mat");


% a

mean_y_train_all = mean(y_train)
N = size(y_val)(1);
mean_y_val = mean(y_val)
mean_std_y_val = std(y_val)/sqrt(N)
mean_y_train = mean(y_train(1:N,:))
mean_std_y_train = std(y_train(1:N,:))/sqrt(N)

% mean_y_train_all =   -9.2497e-15
% mean_y_val = -0.21601
% mean_std_y_val =  0.012904
% mean_y_train = -0.44248
% mean_std_y_train =  0.011927

% the locations are not independently and identically distributed,
%  1, the y_train is not iid, the mean over all data is 0, and it lies
% far outside of the 2 std of the mean over the first 5785 points.

% b




