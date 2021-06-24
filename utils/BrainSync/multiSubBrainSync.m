clc;
clear all;
%% load data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/network_label_259.mat')

direction = 1; % 1 is LR, 2 is RL

% LR data
if direction == 1
    load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_WM_NOGSR.mat')
    load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_WM_LR.mat')

%     load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_LANGUAGE_NOGSR.mat')
%     load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_language_LR.mat')
    missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
    signal_LR(missing_nodes, :, :) = [];

    % get the visual roi
%     signal_LR = signal_LR(network_label, :, :);

    data = signal_LR;
    task_order = 'Tl Bd Fa Tl Bd Pl Fa Pl';
    n_regions = size(data, 1);
    n_frames = size(data, 2);
    n_subs = size(data, 3);
end

% RL data
if direction == 2
    load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_RL_WM_NOGSR.mat')
    load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_WM_RL.mat')

    missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
    signal_RL(missing_nodes, :, :) = [];

    % get the visual roi
%     signal_RL = signal_RL(network_label>=6, :, :);

    data = signal_RL;
    task_order = 'Bd Fa Tl Bd Pl Fa Tl Pl';
    n_regions = size(data, 1);
    n_frames = size(data, 2);
    n_subs = size(data, 3);
end


%% load data
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/network_label_259.mat')
% 
% % LR data
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_WM_NOGSR.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_WM_LR.mat')
% 
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% signal_LR(missing_nodes, :, :) = [];
% 
% % get the visual roi
% %     signal_LR = signal_LR(network_label, :, :);
% 
% true_label_LR = true_label;
% task_order_LR = 'Tl Bd Fa Tl Bd Pl Fa Pl';
% 
% % RL data
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_RL_WM_NOGSR.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_WM_RL.mat')
% 
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% signal_RL(missing_nodes, :, :) = [];
% 
% % get the visual roi
% %     signal_RL = signal_RL(network_label>=6, :, :);
% 
% data = cat(2, signal_LR(:, 9:end, :), signal_RL(:, 9:end, :));
% task_order_RL = 'Bd Fa Tl Bd Pl Fa Tl Pl';
% task_order = 'Tl Bd Fa Tl Bd Pl Fa Pl Bd Fa Tl Bd Pl Fa Tl Pl';
% true_label = [true_label_LR(9:end); true_label(9:end)];
% 
% n_regions = size(data, 1);
% n_frames = size(data, 2);
% n_subs = size(data, 3);


%% normalize data, make each region have zero mean, uniform norm
for i_sub = 1 : n_subs
    [normed_signal, mean_vector, norm_vector] = normalizeData(data(:, :, i_sub)');
end

% while n_subs > 260
%     disp(n_subs)
%     n_subs_r = int32(n_subs / 2) - 1;
%     aligned_mean = zeros(n_regions, n_frames, n_subs_r);
%     for i_sub = 2 : n_subs_r
%         [y2, ~] = brainSync(data(:, :, (i_sub-1)*2+1)', data(:, :, (i_sub-1)*2+2)');
%         aligned_mean(:, :, i_sub) = (data(:, :, (i_sub-1)*2+1) + y2') / 2;
%     end
%     n_subs = n_subs_r;
%     data = aligned_mean;
% end    
aligned_mean = zeros(n_regions, n_frames, n_subs);
aligned_mean(:, :, 1) = data(:, :, 1);
for i_sub = 2 : n_subs
        [y2, ~] = brainSync(data(:, :, 1)', data(:, :, i_sub)');
        aligned_mean(:, :, i_sub) = y2';
end
data = aligned_mean;
for i = 1 : n_subs
    data(:, :, i) = zscore(data(:, :, i), 0, 2);
end



%% diffusion map
n_d = 10; % reduced dimension
dParams_1.kNN = 140; % num of nearest neighbors when computing affinity matrix
dParams_1.self_tune = 0; % choose the bandwidth
% dParams_1.dist_type = 'cosine';
dParams_1.dist_type = 'euclidean';

dParams_diffusion_1.normalization = 'markov';
dParams_diffusion_1.t = 1;
dParams_diffusion_1.verbose = 0;
dParams_diffusion_1.plotResults = 0;
dParams_diffusion_1.maxInd = n_d+1;

[K, nnData] = calcAffinityMat(data, dParams_1);
[diffusion_map, Lambda, Psi, Ms, Phi, K_rw] = calcDiffusionMap(K, dParams_diffusion_1);
figure;imagesc(K)

s = true_label;
% s = 1:n_frames;
c = 25*ones(n_frames, 1);

% figure;scatter3(diffusion_map(1,:),diffusion_map(2,:),diffusion_map(3,:), c, s, 'filled')
figure;scatter3(diffusion_map(2,:),diffusion_map(3,:),diffusion_map(4,:), c, s, 'filled')

figure;imagesc(diffusion_map')

% as individual lines
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k'];

figure;
plot(3,3);
for i = 1:7
    subplot(3,3,i);
    imagesc(0,-5,repmat(true_label', 11, 1), 'AlphaData', 0.7)
%     title(task_order)
    colormap(gray)
    hold on;
    coordinate = i;
    p = plot(diffusion_map(coordinate, :)'*50, colors(coordinate));
    p.LineWidth = 1.5;
    xlim([1,n_frames]);
    ylim([-5, 5])
%     set(gcf, 'color', 'none');
%     set(gca, 'color', 'none');
end

n_c = 4;
idx = kmeans(diffusion_map(1:5, :)', n_c,  'replicates', 1000, 'display','final');
figure;
imagesc(repmat(true_label', n_c, 1), 'AlphaData', 0.7)
% title(task_order)
hold on;
plot(idx, 'r')

cluster_response = zeros(n_regions, n_c);
for i = 1 : n_c
    cluster_response(:, i) = mean(reshape(data(:, idx==i, :), n_regions, []),2);
end



%% 2rDM
%% parameters configuration for the diffusion map
n_d = 10; % reduced dimension
dParams_1.kNN = 140; % num of nearest neighbors when computing affinity matrix
dParams_1.self_tune = 0; % choose the bandwidth
% dParams_1.dist_type = 'cosine';
dParams_1.dist_type = 'euclidean';

dParams_diffusion_1.normalization = 'markov';
dParams_diffusion_1.t = 1;
dParams_diffusion_1.verbose = 0;
dParams_diffusion_1.plotResults = 0;
dParams_diffusion_1.maxInd = n_d+1;

%% first round of diffusion map computation
embed = zeros(size(data, 2), n_subs*n_d);
for i_sub = 1 : n_subs
    data_ind = data(:, :, i_sub);

    [K, nnData] = calcAffinityMat(data_ind, dParams_1);
    [diffusion_map, Lambda, Psi, Ms, Phi, K_rw] = calcDiffusionMap(K, dParams_diffusion_1);
    embed(:, (i_sub-1)*n_d+1:(i_sub)*n_d) = diffusion_map(1:n_d, :)';
end

%% second round of diffusion map computation
dParams_2 = dParams_1;
dParams_2.dist_type = 'euclidean';

dParams_diffusion_2 = dParams_diffusion_1;
[K, nnData] = calcAffinityMat(embed', dParams_2);
% visualize the population-based affinity matrix
figure;imagesc(K)
[diffusion_map, Lambda, Psi, Ms, Phi, K_rw] = calcDiffusionMap(K, dParams_diffusion_2);

%% visualize the diffusion map

% s = 1 : n_frames;
s = true_label;
c = 25*ones(n_frames, 1);

% figure;scatter3(diffusion_map(1,:),diffusion_map(2,:),diffusion_map(3,:), c, s, 'filled')
figure;scatter3(diffusion_map(2,:),diffusion_map(5,:),diffusion_map(4,:), c, s, 'filled')

figure;imagesc(diffusion_map')

% as individual lines
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k'];

figure;
plot(3,3);
for i = 1:7
    subplot(3,3,i);
    imagesc(0,-5,repmat(true_label', 11, 1), 'AlphaData', 0.7)
    title(task_order)
    colormap(gray)
    hold on;
    coordinate = i;
    p = plot(diffusion_map(coordinate, :)'*50, colors(coordinate));
    p.LineWidth = 1.5;
    xlim([1,n_frames]);
    ylim([-5, 5])
%     set(gcf, 'color', 'none');
%     set(gca, 'color', 'none');
end

%% k-means clustering on diffusion map coordinates
n_c = 4;
idx = kmeans(diffusion_map(2:5, :)', n_c,  'replicates', 1000, 'display','final');
figure;
imagesc(repmat(true_label', n_c, 1), 'AlphaData', 0.7)
title(task_order)
hold on;
plot(idx, 'r')
title('2rDM kmeans')
cluster_response = zeros(n_regions, n_c);
for i = 1 : n_c
    cluster_response(:, i) = mean(reshape(data(:, idx==i, :), n_regions, []),2);
end
