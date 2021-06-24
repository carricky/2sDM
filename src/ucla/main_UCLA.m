% load network definition
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map268.mat')
map268(map268==6) = 5;
map268(map268==7) = 5;
network_name = {'MF', 'FP', 'DMN', 'Motor', 'Visual', 'Subcortical', 'Cerebellum'};

% load data
[ucla_task, ~] = data_label_generation(1);
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% ucla_task(missing_nodes, :, :) = [];

% load labels, subject type and specific colormap
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_labels.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/idx199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/colormaps/cmap12.mat')

% parse the size
n_rg = size(ucla_task, 1);

% set the subject and time
sub_idx = idx_control(1:20);
% sub_idx = idx_schz(1:20);

n_sub = numel(sub_idx);
tp_idx = 1 : 1009;
n_tp = size(ucla_task, 2);

% final config
rng(665)
vmin = -0.5;
vmax = 11.5;

%% config the parameters
k = 150;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;
configDiffParams1.t = 1;
configDiffParams1.normalization='lb';
configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;
n_dim = 7;

%% 2sDM for control
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(ucla_task(:, :, ...
    sub_idx), n_dim, configAffParams1, configAffParams2, ...
    configDiffParams1, configDiffParams2);

figure;
scatter3(dm(1, :), dm(2, :), dm(3, :), 40, all_labels, 'filled');
colormap(cmap12)
caxis([vmin, vmax])

% figure; imagesc(dm);

figure;
temp_label = all_labels;
temp_label(temp_label<1 | temp_label>2) = 0;
scatter3(dm(1, :), dm(2, :), dm(3, :), 40, temp_label, 'filled');
colormap(cmap12)
caxis([vmin, vmax])

k = 4;
IDX = kmeans(dm(:, :)', k, 'Replicates', 100);

%% PC calculation
% create temporal label
task_idx = [[1,242]; [243, 510]; [511, 801]; [802, 1009]];
temporal_label = zeros(1, n_tp);
for i_task = 1 : size(task_idx, 1)
    temporal_label(task_idx(i_task, 1):task_idx(i_task, 2)) = i_task;
end

[pc_global, mc_global, pc_region, mc_region] = computeDynamicNetworkMeasure(...
    ucla_task(:, tp_idx, sub_idx), map268, 5, temporal_label);

% examine if any subnetwork is interesting
count = 1;
figure;
for i_network = [1,2,3,4,5,8,9]
    subplot(2,4,count)
    pc_mean = mean(pc_region(map268==i_network, :));
    plot_idx = (pc_mean~=0);
    temp_corr = corr(dm(:, plot_idx)', pc_mean(plot_idx)')
    [~, order] = sort(abs(temp_corr), 'descend');
%     scatter3(dm(1, plot_idx), dm(2, plot_idx), dm(3, plot_idx), 25, pc_mean(plot_idx), 'filled');
    scatter(dm(order(1), plot_idx), dm(order(2), plot_idx), ...
        15, pc_mean(plot_idx), 'filled');
    colormap('jet')
    title(network_name{count})
    xlabel(sprintf('coord%d, corr=%.3f', order(1), temp_corr(order(1))))
    ylabel(sprintf('coord%d, corr=%.3f', order(2), temp_corr(order(2))))
    caxis([0.822, 0.842])
    count = count + 1;
end
plot_idx = (pc_global~=0);
temp_corr = corr(dm(:, plot_idx)', pc_global(plot_idx)')
[~, order] = sort(abs(temp_corr), 'descend');
subplot(2,4,count)
scatter(dm(order(1), plot_idx), dm(order(2), plot_idx), 15, ...
    pc_global(plot_idx), 'filled');
colormap('jet')
title('Whole Brain')
xlabel(sprintf('coord%d, corr=%.3f', order(1), temp_corr(order(1))))
ylabel(sprintf('coord%d, corr=%.3f', order(2), temp_corr(order(2))))
caxis([0.822, 0.842])

% examine specific dimensions
figure;
% dim_spec = [2,3,4];
dim_spec = [1,2,3];
scatter3(dm(dim_spec(1), plot_idx), dm(dim_spec(2), plot_idx),...
    dm(dim_spec(3), plot_idx), 25, pc_global(plot_idx), 'filled');
colormap('jet')
caxis([0.822, 0.842])
figure;
scatter3(dm(dim_spec(1), plot_idx), dm(dim_spec(2), plot_idx), ...
    dm(dim_spec(3), plot_idx), 25, all_labels(plot_idx), 'filled');
colormap(cmap12)
caxis([vmin, vmax])

% exmaine box plot
count = 1;
figure;
for i_network = [1,2,3,4,5,8,9]
    subplot(2,4,count)
    pc_mean = mean(pc_region(map268==i_network, :));
    boxplot(pc_mean(plot_idx), temporal_label(plot_idx))
    title(network_name{count})
%     caxis([0.85, 0.88])
    count = count + 1;
end
subplot(2,4,count)
boxplot(pc_global(plot_idx), temporal_label(plot_idx))
title('Whole Brain')

%% test PCA
% [ucla_task, ~] = data_label_generation(1);
% 
% embed = zeros(n_tp, n_sub*n_dim);
% for i_sub = 1 : n_sub
%     disp(i_sub/n_sub)
%     data_ind = ucla_task(:, :, i_sub);
%     [~, score] = pca(data_ind', 'NumComponents', n_dim);
%     embed(:, (i_sub-1)*n_dim+1:(i_sub)*n_dim) = score;
% end
% [~, score] = pca(embed, 'NumComponents', n_dim);
% 
% temp_corr = corr(score, pc_global')
% [~, order] = sort(abs(temp_corr), 'descend');
% score = score';
% 
% figure;
% dim_spec = [order(1),order(2),order(3)];
% % dim_spec = [1, 2, 3];
% scatter3(score(dim_spec(1), plot_idx), score(dim_spec(2), plot_idx), ...
%     score(dim_spec(3), plot_idx), 25, pc_global(plot_idx), 'filled');
% colormap('jet')
% xlabel(sprintf('coord%d, corr=%.3f', dim_spec(1), temp_corr(dim_spec(1))))
% ylabel(sprintf('coord%d, corr=%.3f', dim_spec(2), temp_corr(dim_spec(2))))
% 
% figure;
% dim_spec = [order(1),order(2),order(3)];
% % dim_spec = [1, 2, 3];
% scatter3(score(dim_spec(1), plot_idx), score(dim_spec(2), plot_idx), ...
%     score(dim_spec(3), plot_idx), 25, all_labels(plot_idx), 'filled');
% colormap(cmap12)
% caxis([vmin, vmax])   
% xlabel(sprintf('coord%d, corr=%.3f', dim_spec(1), temp_corr(dim_spec(1))))
% ylabel(sprintf('coord%d, corr=%.3f', dim_spec(2), temp_corr(dim_spec(2))))

%% embed subjects
% % as we can embed and reveal different participation coefficients, schz subjects
% % may have higher PC, why not embed them together and see if we can seperate
% % them
% [ucla_task, ~] = data_label_generation(1);
% % for i_sub = 1 : size(ucla_task, 3)
% %     ucla_task(:, :, i_sub) = zscore(ucla_task(:, :, i_sub), 0, 2);
% % end
% ucla_task_sub = permute(ucla_task, [1, 3, 2]);
% k = 40;
% configAffParams1.dist_type = 'euclidean';
% configAffParams1.kNN = k;
% configAffParams1.self_tune = 0;
% configDiffParams1.t = 1;
% configDiffParams1.normalization='lb';
% configAffParams2 = configAffParams1;
% configDiffParams2 = configDiffParams1;
% n_dim = 7;
% 
% % sub_idx = [idx_control, idx_schz];
% sub_idx = 1 : 199;
% [dm_sub, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(ucla_task_sub(:,...
%     sub_idx, :), n_dim, configAffParams1, configAffParams2, ...
%     configDiffParams1, configDiffParams2);
% 
% c_sub = zeros(numel(sub_idx), 1);
% c_sub(1:numel(idx_control)) = 1; 
% figure; 
% scatter3(dm_sub(1, :), dm_sub(2, :), dm_sub(3, :), 40, sub_type, 'filled');
% colormap('jet')
