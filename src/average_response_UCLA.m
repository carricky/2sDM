%% this script generates brain response plot of each cluster in k-means clustering from 2sDM, the dm should be generated first
ucla_task = data_label_generation(1); % 1 represents z-score is used

% load labels, subject type and specific colormap
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_labels.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/idx199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/cmap12.mat')

% set the subject number and time
sub_idx_control = idx_control(1:44);

% algorithm unrelated unconfig
rng(665)
vmin = -0.5;
vmax = 11.5;

%% generate embedding and kmeans
configAffParams1.dist_type = 'euclidean';
configAffParams1.self_tune = 0;
configDiffParams1.t = 1;
configDiffParams1.normalization='lb';
configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;
n_dim = 7;

configAffParams1.kNN = 130;
configAffParams2.kNN = 130;

dm_con = calc2sDM(ucla_task(:, :, sub_idx_control), n_dim, configAffParams1, ...
    configAffParams2, configDiffParams1, configDiffParams2);

n_cluster = 4;
[kmeans_idx_con, C, ~, D] = kmeans(dm_con(:, :)', n_cluster);

figure;
subplot(1,2,1)
scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, all_labels, 'filled');
colormap(cmap12)
caxis([vmin, vmax])

subplot(1,2,2)
scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, kmeans_idx_con, 'filled');
colormap(cmap12)
caxis([vmin, vmax])

%% find representative cluster centers
% fix_id = find(dm_con(1,:)<-0.0288 & dm_con(2,:)<-0.0125);
% low_id = find(dm_con(1,:)>0.0283 & dm_con(2,:)<-0.0137);
% high_id = find(dm_con(1,:)>5.5064e-04 & dm_con(2,:)>0.0116);
% cluster_id = [fix_id, low_id, high_id];
% 
% % plot the cluster center on the kmeans plot
% c = kmeans_idx_con;
% c(cluster_id) = 6;
% subplot(2,2,3);scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, c, 'filled');
% colormap(cmap12)
% caxis([vmin,vmax])
% 
% % get a k-corr plot to determine a good k
% k_list = 1:5:600;
% [D, I] = pdist2(dm_con', dm_con', 'Euclidean', 'Smallest', 600);
% 
% % generate the average response
% all_response = zeros(268, 3);
% figure;
% for i_c = 1 : 3
%     near_idx = I(:, cluster_id(i_c))+1009;
%     corr_list = [];
%     % generate the neighbor response
%     all_response(:, i_c) = mean(mean(ucla_task(:, cluster_id(i_c), sub_idx_control), 3), 2);
% end


%% Use k-means cluster centers
high_id = find(D(:,1)==min(D(:,1)));
low_id = find(D(:,2)==min(D(:,2)));
fix_id = find(D(:,3)==min(D(:,3)));
cluster_id = [fix_id, low_id, high_id];

% generate the average response
all_response = zeros(268, n_cluster);
for i_c = 1 : n_cluster
    all_response(:, i_c) = mean(mean(ucla_task(:, cluster_id(i_c), sub_idx_control), 3), 2);
end

c = kmeans_idx_con;
c(cluster_id) = 6;
figure;
scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, c, 'filled');
colormap(cmap12)
caxis([vmin,vmax])

%% Use 10 nearest points to the k-means cluster centers
D_sorted = sort(D(:, 1));
fix_id = find(D(:,1)<=D_sorted(10));

D_sorted = sort(D(:, 2));
low_id = find(D(:,2)<=D_sorted(10));

D_sorted = sort(D(:, 3));
high_id = find(D(:,3)<=D_sorted(10));

cluster_id = [fix_id, low_id, high_id];

% generate the average response
all_response = zeros(268, 3);
for i_c = 1 : 3
    % generate the neighbor response
    all_response(:, i_c) = mean(mean(ucla_task(:, cluster_id(:, i_c), sub_idx_control), 3), 2);
end

%% !!Experimental Section
% correlate response with HCP dataset clustering centroid
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/extension/all_response.mat')
all_response = squeeze(all_response(:, 1, :));
% load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/hcp/all_response_task.mat')


ucla_task_avg = mean(ucla_task(:, :, sub_idx_control), 3);

corr_hcp_ucla = corr(ucla_task_avg, all_response);

figure;
for i_c = 1 : 4
    subplot(2,2,i_c)
    scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, corr_hcp_ucla(:, i_c), 'filled');
    colormap('jet')
end