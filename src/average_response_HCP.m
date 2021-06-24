%% load data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% choose less subjects for quicker computation
sub_range = 1:30;
train_range = 1:3020;

data = data(:, :, sub_range);

train_data = data(:, train_range, :); % task data to generate manifold
num_s = size(train_data, 3); % number of subjects
num_t_train = size(train_data, 2); % length of training data
num_r = size(train_data, 1); % number of regions
num_t_all = num_t_test+num_t_train; % total length of time

missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
missing_nodes_binary = zeros(268, 1);
missing_nodes_binary(missing_nodes) = 1;
%% config the parameters
k = 500;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 7;

%% generate training embedding
% embed is the n_t*(n_sub*n_d) first round flatten matrix, dm is the final embedding
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2); 

s = 20*ones(num_t_train, 1);
c = true_label_all(train_range);
vmin = -0.5;
vmax = 17.5;

figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])

subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])

%% clustering the data
n_cluster = 4;
rng(665)
[IDX, C, ~, D]= kmeans(dm(:, :)', n_cluster);
c = true_label_all(1:size(dm,2));
s = 20*ones(size(dm,2), 1);

figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(4, :), s, IDX, 'filled');
colormap(cmap18)
caxis([vmin,vmax])
subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])

%% find representative cluster centers
low_id = find(D(:,1)==min(D(:,1)));
fix_id = find(D(:,2)==min(D(:,2)));
cue_id = find(D(:,3)==min(D(:,3)));
high_id = find(D(:,4)==min(D(:,4)));
cluster_id = [fix_id, cue_id, low_id, high_id];

% plot the cluster center on the kmeans plot
s = 25*ones(size(dm,2), 1);
c = IDX;
c(cluster_id) = 6;
s(cluster_id) = 200;
subplot(2,2,3);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])

%% generate the average response
all_response = zeros(268, n_cluster);
for i_c = 1 : n_cluster
    all_response(~missing_nodes_binary, i_c) = mean(mean(train_data(:, cluster_id(i_c), :), 3), 2);
end