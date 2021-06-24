theta = 0:1/2000:1;
x = rand(1, 2001);
s = [pi*(1+3*theta).*cos(pi*(1+3*theta)); 50*x; pi*(1+3*theta).*sin(pi*(1+3*theta))];

figure(1);
scatter3(s(1, :), s(2, :), s(3, :), 20, theta, 'filled')

n_iter = 3;
n_dim = 20;
k = 100;
% time_idx = [1:1553, 3021:3620];

x = all_task(:, time_idx, sub);

configAffParams.dist_type = 'euclidean';
configAffParams.kNN = k;
configAffParams.self_tune = 0;

configDiffParams.t = 1;
configDiffParams.normalization='lb';
configDiffParams.maxInd = n_dim+1;

[dm, K, lambda, sigma] = IDM(s, n_dim, n_iter, configAffParams, configDiffParams, 0);

for i_iter = 1 : n_iter
    figure(2);
    subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
    scatter3(dm(:, 1, i_iter), dm(:, 2, i_iter), dm(:, 3, i_iter), 20, theta, 'filled');
    title(sprintf('iter %d', i_iter))
    
    figure(3);
    subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
    imagesc(K(:, :, i_iter))
    title(sprintf('iter %d', i_iter))
    
    figure(4);
    subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
    imagesc(dm(:, :, i_iter))
    title(sprintf('iter %d', i_iter))
end