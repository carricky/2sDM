load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/WM_accuracy_RT390.mat')

% choose less subjects for quicker computation
data = data(:, :, 1:390);

% leftout WM task
n_tp_test = 387;
test_range = 1 : n_tp_test;
train_range = n_tp_test+1:3020;
test_data = data(:, test_range, :);

train_data = data(:, train_range, :);
n_sub = size(train_data, 3);
n_tp_all = size(train_data, 2);
n_region = size(train_data, 1);

test_label = true_label_all(test_range);

%% config the parameters
k = 500;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_dim = 7;

%% generate training embedding
[dm_train, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data,...
    n_dim, configAffParams1, configAffParams2, configDiffParams1,...
    configDiffParams2);

%% generate all embedding
dm_whole = calc2sDM(data(:, 1:3020, 1:390), n_dim, configAffParams1, configAffParams2, ...
    configDiffParams1, configDiffParams2);

% clustering
rng(665)
n_cluster = 4;
kmeans_map = [4, 2, 1, 3];
% kmeans_map = [1, 2, 3, 4];

kmeans_idx_whole = kmeans(dm_whole(1:4, :)', n_cluster, ...
    'Replicates', 100);
for i_sub = 1 : n_cluster
    kmeans_idx_whole(kmeans_idx_whole==i_sub)=kmeans_map(i_sub)+n_cluster;
end
kmeans_idx_whole = kmeans_idx_whole - n_cluster;

% plot
figure;
scatter3(dm_whole(1, :), dm_whole(2, :), dm_whole(3, :), 20, kmeans_idx_whole, 'filled');
colormap(cmap4)
caxis([0.5, 5.5])

figure;
c = true_label_all(1:3020);
scatter3(dm_whole(1, :), dm_whole(2, :), dm_whole(3, :), 20, c, 'filled');
colormap(cmap10)
caxis([-0.5,9.5])

%% Build brain states with the original embedding
% rng(665)
% n_cluster = 4;
% kmeans_map = [4, 2, 1, 3];
% % kmeans_map = [1, 2, 3, 4];
% 
% kmeans_idx = kmeans(dm_train(1:4, :)', n_cluster, ...
%     'Replicates', 100);
% for i_sub = 1 : n_cluster
%     kmeans_idx(kmeans_idx==i_sub)=kmeans_map(i_sub)+n_cluster;
% end
% kmeans_idx = kmeans_idx - n_cluster;

kmeans_idx = kmeans_idx_whole(train_range);

%% plot
figure;
scatter3(dm_train(1, :), dm_train(2, :), dm_train(3, :), 20, kmeans_idx, 'filled');
colormap(cmap4)
caxis([0.5, 5.5])

figure;
c = true_label_all(train_range);
scatter3(dm_train(1, :), dm_train(2, :), dm_train(3, :), 20, c, 'filled');
colormap(cmap10)
caxis([-0.5,9.5])

%% generate extension embedding single match
all_trans_mat_rs = zeros(n_sub, n_cluster, n_cluster);
all_stationary_p_0back = zeros(n_sub, n_cluster);
all_stationary_p_2back = zeros(n_sub, n_cluster);

figure;
tic;
for i_sub = 1 : n_sub
    disp(i_sub)
    %     [K] = calcAffinityMat2(wm_data(:,:,i), data(:,:,i), k, sigma1(i));
    data_single = [test_data(:,:,i_sub), train_data(:,:,i_sub)];
    
    % laplacian pyramid
%     configAffParams3 = configAffParams1;
%     configAffParams3.kNN = 500;
%     configAffParams3.self_tune = 0;
%     configAffParams3.n_sub = n_sub;
%     dm_all = zeros(n_tp_total, n_dim);
%     % compute dXX only once to speed up
%     [dm_all(tp_idx_test, 1), l(1), dXX, dXY, indsXX, indsXY] = ...
%         calcLaplcianPyramidExtension(configAffParams3, x_tilda_all(:, ...
%         tp_idx_train), (dm_train(1,:)./lambda2(1))', x_tilda_all(:, tp_idx_test));
%     for i = 2 : n_dim
%         [dm_all(tp_idx_test, i), l(i)] = calcLaplcianPyramidExtension(...
%             configAffParams3, test_data(:,:,i_sub), ...
%             (dm_train(i,:)./lambda2(i))', train_data(:,:,i_sub), dXX, dXY, indsXX, indsXY);
%     end
    
    % nystrom
    configAffParams3 = configAffParams1;
    configAffParams3.kNN = 500;
    configAffParams3.sig = sigma1(i_sub);
    [K] = calcAffinityMat(data_single, configAffParams3);
    K = K(test_range, train_range);
    K = K./sum(K, 2);
    psi2 = K*dm_train'./lambda2';
    dm_all = [psi2', dm_train];
        
%     c = true_label_all(1:3020);
%     c(train_range) = 10;
%     scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), 20, c, 'filled');
%     colormap(cmap10)
%     caxis([-0.5,9.5])
%     drawnow
%     pause
    
    [~, I] = pdist2(dm_train(1:3, :)', dm_all(1:3, test_range)', 'euclidean',...
        'Smallest', 1);
    wm_kmeans_idx = kmeans_idx(I);
    
    trans_mat_0back = zeros(n_cluster, n_cluster);
    trans_mat_2back = zeros(n_cluster, n_cluster);
    for j_tp = 1 : numel(test_range)-1
        if test_label(j_tp) == 2 && test_label(j_tp+1) == 2
            c1 = wm_kmeans_idx(j_tp);
            c2 = wm_kmeans_idx(j_tp+1);
            trans_mat_0back(c1, c2) = trans_mat_0back(c1, c2) + 1;
        end
        if test_label(j_tp) == 3 && test_label(j_tp+1) == 3
            c1 = wm_kmeans_idx(j_tp);
            c2 = wm_kmeans_idx(j_tp+1);
            trans_mat_2back(c1, c2) = trans_mat_2back(c1, c2) + 1;
        end
    end
    
    trans_mat_0back = trans_mat_0back./ sum(trans_mat_0back, 2);
    trans_mat_0back(isnan(trans_mat_0back)) = 0;
    [temp, ~] = eigs(trans_mat_0back');
    stationary_p_0back = temp(:, 1) / sum(temp(:, 1));
    
    trans_mat_2back = trans_mat_2back./ sum(trans_mat_2back, 2);
    trans_mat_2back(isnan(trans_mat_2back)) = 0;
    [temp, ~] = eigs(trans_mat_2back');
    stationary_p_2back = temp(:, 1) / sum(temp(:, 1));
    
    all_stationary_p_0back(i_sub, :) = stationary_p_0back;
    all_stationary_p_2back(i_sub, :) = stationary_p_2back;
       
%     % 0-back
%     n_0back = numel(wm_kmeans_idx(test_label==2));
%     for j = 1 : n_cluster
%         all_stationary_p_0back(i_sub, j) = sum(wm_kmeans_idx(test_label==2)==j)/n_0back;
%     end
%     % 2-back
%     n_2back = numel(wm_kmeans_idx(test_label==3));
%     for j = 1 : n_cluster
%         all_stationary_p_2back(i_sub, j) = sum(wm_kmeans_idx(test_label==3)==j)/n_2back;
%     end
    
end
toc;
%% Validate the extension
idx = ~isnan(measures(:, 2));

figure;
plot(all_stationary_p_0back(idx, 1), measures(idx, 1), '.')

[r, p] = corr(all_stationary_p_0back(idx, :), measures(idx, :))
[r, p] = corr(all_stationary_p_2back(idx, :), measures(idx, :))

figure;
subplot(1,2,1)
boxchart(all_stationary_p_0back)
ylim([0, 1])
subplot(1,2,2)
boxchart(all_stationary_p_2back)
ylim([0, 1])


