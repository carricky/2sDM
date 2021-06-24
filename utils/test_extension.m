%% make points
x_range_train = -2:0.4:2;
y_range_train = -2:0.4:2;
[x_train, y_train] = meshgrid(x_range_train, y_range_train);
z_train = sin(sqrt(x_train.^2+y_train.^2));

figure;
subplot(2,2,1);
imagesc(x_range_train, y_range_train, z_train);
title('training data')

% laplacian pyramid extension
x_range_test = -3:0.01:3;
y_range_test = -3:0.01:3;
[x_test, y_test] = meshgrid(x_range_test, y_range_test);
data_train = [x_train(:), y_train(:)];
data_test = [x_test(:), y_test(:)];
[z_ext, ll] = calcLaplcianPyramidExtension([], data_train', z_train(:), data_test');
z_ext = reshape(z_ext, numel(x_range_test), numel(y_range_test));
subplot(2,2,2);
imagesc(x_range_test, y_range_test, z_ext);
title('extension')

% ground truth
z_truth = sin(sqrt(x_test.^2+y_test.^2));

subplot(2,2,4);
imagesc(x_range_test, y_range_test, z_truth);
title('ground truth')

% difference
subplot(2,2,3);
imagesc(x_range_test, y_range_test, abs(z_ext-z_truth));
title('extension')