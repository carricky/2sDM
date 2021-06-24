% choose less subjects for quicker computation
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/idx199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/rest199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_labels.mat')
load('/Users/siyuangao/Working_Space/fmri/data/colormaps/cmap12.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap6.mat')

data = data_label_generation(1);
idx_control = idx_control(1:77);
all_task = data(:, :, idx_control);
all_rest = all_signal(:, :, idx_control);

n_tp_task = size(all_task, 2);
n_tp_rest = size(all_rest, 2);
n_tp_all = n_tp_task + n_tp_rest;
n_sub = size(all_task, 3);
n_region = size(all_task, 1);

% algorithm unrelated unconfig
rng(665)
vmin = -0.5;
vmax = 11.5;

%% config the parameters
k = 130;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 7;

%% generate training embedding
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(all_task, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

s = 40*ones(n_tp_task, 1);
c = all_labels;
figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap12)
caxis([vmin, vmax])
subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap12)
caxis([vmin, vmax])

%% cluster the tasks
k = 4;
IDX = kmeans(dm(:, :)', k);

%% generate extension embedding
dwell_time_list = zeros(n_sub, 4);
IDX_list_all = zeros(n_sub, n_tp_rest);

train_range = 1 : n_tp_task;
test_range = n_tp_task+1 : n_tp_all;

k_ext = 150;
ref_sub = 1;
psi1 = zeros(n_tp_rest, n_d*n_sub);
configAffParams3 = configAffParams1;
configAffParams3.kNN = k_ext;
all_rest_sync = zeros(n_region, n_tp_rest, n_sub);

for i = 1 : n_sub
    disp(i)
    data_ind = zeros(n_region, n_tp_all);
    data_ind(:, train_range) = all_task(:, :, i);
    data_ind(:, test_range) = all_rest(:, :, i);
    if i ~= ref_sub
        [Y2, R] = brainSync(all_rest(:, :, ref_sub)', data_ind(:, test_range)');
        data_ind(:, test_range) = Y2';
    end
    all_rest_sync(:, :, i) = data_ind(:, test_range);
    configAffParams3.sig = sigma1(i);
    [K] = calcAffinityMat(data_ind, configAffParams3);
    K = K(test_range, train_range);
    K = K./sum(K, 2);
    psi1(:, (i-1)*n_d+1:i*n_d) = K*embed(:, (i-1)*n_d+1:i*n_d)./lambda1(:, i)';
end

% second round embedding

embed_all = zeros(n_d*n_sub, n_tp_all);
embed_all(:, train_range) = embed';
embed_all(:, test_range) = psi1';
configAffParams3.sig = sigma2;
[K, ~] = calcAffinityMat(embed_all, configAffParams3);
K = K(test_range, train_range);
K = K./sum(K, 2);
psi2 = K*dm'./lambda2';

dm_all = zeros(n_d, n_tp_all);
dm_all(:, train_range) = dm;
dm_all(:, test_range) = psi2';

% [D_ext, I_ext] = pdist2(dm_all(:, train_range)', dm_all(:, test_range)', 'Euclidean', 'Smallest', 10);
% IDX_list = zeros(size(I_ext, 2), 1);
% for i = 1 : size(I_ext, 2)
%     IDX_temp = mode(IDX(I_ext(:, i)));
%     IDX_list(i) = IDX_temp;
% end
% for i = 1 : 4
%     dwell_time_list(ref_sub, i) = (sum(IDX_list==i)/numel(IDX_list));
% end
% IDX_list_all(ref_sub, :) = IDX_list;
% disp(dwell_time_list(ref_sub, :));

% plot the extension
c = 5*ones(n_tp_all, 1);
c(train_range) = IDX;

figure;
scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), 40, c, 'filled');
colormap(cmap6)
caxis([0.5, 6.5])

%% identify extended rs activation that are similar to task
D_all = pdist2(dm_all', dm_all', 'euclidean');
D_task_rest = D_all(train_range, test_range);
for k_rest = 10
    [~, n_nearest_rest_index] = sort(D_task_rest, 2);
    n_nearest_rest_index = n_nearest_rest_index(:, 1:k_rest);
    all_task_avg = mean(all_task, 3);
    all_rest_avg = mean(all_rest_sync, 3);
    
    all_task_rest_corr = zeros(n_tp_task, 1);
    all_task_near_rest_response = zeros(n_tp_task, n_region);
    for i_tp = 1 : n_tp_task
        all_task_near_rest_response(i_tp, :) = mean(all_rest_avg(:, n_nearest_rest_index(i_tp, :)), 2);
        all_task_rest_corr(i_tp) = corr(all_task_near_rest_response(i_tp, :)', all_task_avg(:, i_tp));
    end
    figure;
    scatter3(dm(1, :), dm(2, :), dm(3, :), 40, all_task_rest_corr, 'filled');
    disp(k_rest)
    disp(mean(all_task_rest_corr))
    disp(max(all_task_rest_corr))
end

g = ginput(3);
[~, g_index] = pdist2(dm(1:2, :)', g, 'euclidean', 'Smallest', 1);
avg_task_response = all_task_avg(:, g_index);
avg_rest_response = all_task_near_rest_response(g_index, :)';
disp(corr(avg_rest_response, avg_task_response))

%% clustering the data
c = label(1:size(dm,2));
s = 20*ones(size(dm,2), 1);

figure;subplot(2,1,1);scatter3(dm(1, :), dm(2, :), dm(4, :), s, IDX, 'filled');
subplot(2,1,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])