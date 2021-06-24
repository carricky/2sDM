%% load data
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map268.mat')
map268(map268==6) = 5;
map268(map268==7) = 5;
network_name = {'MF', 'FP', 'DMN', 'Motor', 'Visual', 'Subcortical', 'Cerebellum'};

ucla_task = data_label_generation(1); % 1 represents z-score is used
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% ucla_task(missing_nodes, :, :) = [];

task_endtime = [0,242,510,801,1009];

% load labels, subject type and specific colormap
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_labels.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/idx199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/colormaps/cmap12.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')

% set the subject number and time
rng(665)
random_idx1 = zeros(77, 1);
random_idx1(randsample(77, 77)) = 1;
random_idx1 = logical(random_idx1);

sub_idx_control = idx_control(random_idx1');
sub_idx_control2 = idx_control(~random_idx1');

sub_idx_schz = idx_schz(1:44);
tp_idx = 1 : 1009;
n_tp = size(ucla_task, 2);

% algorithm unrelated config
vmin = -0.5;
vmax = 11.5;

stateNames = ["High cog" "Transition" "Fixation" "Low cog"];
taskNames = ["PAMENC" "PAMRET" "SCAP" "TaskSwitch"];

%% config the parameters
configAffParams1.dist_type = 'euclidean';
configAffParams1.self_tune = 0;
configDiffParams1.t = 1;
configDiffParams1.normalization='lb';
configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;
n_dim = 7;

%% 2sDM for control
configAffParams1.kNN = 200;
configAffParams2.kNN = 200;
dm_con = calc2sDM(ucla_task(:, :, sub_idx_control), n_dim, configAffParams1, ...
    configAffParams2, configDiffParams1, configDiffParams2);

figure;
subplot(1,2,1)
scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, all_labels, 'filled');
colormap(gca, cmap12)
caxis([vmin, vmax])

rng(665)
n_cluster = 4;
kmeans_idx_con = kmeans(dm_con(1:3, :)', n_cluster, 'Replicates', 100);

kmeans_idx_con(kmeans_idx_con == 1) = 5;
kmeans_idx_con(kmeans_idx_con == 2) = 1;
kmeans_idx_con(kmeans_idx_con == 5) = 2;


subplot(1,2,2)
scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, kmeans_idx_con, 'filled');
colormap(gca, cmap4)
caxis([0.5, 5.5])

%% 2sDM for control cohort 2
% configAffParams1.kNN = 220;
% configAffParams2.kNN = 220;
% dm_con2 = calc2sDM(ucla_task(:, :, sub_idx_control2), n_dim, configAffParams1, ...
%     configAffParams2, configDiffParams1, configDiffParams2);
% 
% figure;
% subplot(1,2,1)
% scatter3(dm_con2(1, :), dm_con2(2, :), dm_con2(3, :), 40, all_labels, 'filled');
% colormap(gca, cmap12)
% caxis([vmin, vmax])
% 
% rng(665)
% n_cluster = 4;
% kmeans_idx_con2 = kmeans(dm_con2(1:4, :)', n_cluster, 'Replicates', 100);
% 
% subplot(1,2,2)
% scatter3(dm_con2(1, :), dm_con2(2, :), dm_con2(3, :), 40, kmeans_idx_con2, 'filled');
% colormap(gca, cmap6)
% caxis([0.5, 6.5])

%% 2sDM for schz
configAffParams1.kNN = 220;
configAffParams2.kNN = 220;
dm_schz = calc2sDM(ucla_task(:, :, sub_idx_schz), n_dim, configAffParams1, ...
    configAffParams2, configDiffParams1, configDiffParams2);

figure;
subplot(1,2,1)
scatter3(dm_schz(1, :), dm_schz(2, :), dm_schz(3, :), 40, all_labels, 'filled');
colormap(gca, cmap12)
caxis([vmin, vmax])

rng(665)
n_cluster = 4;
kmeans_idx_schz = kmeans(dm_schz(1:3, :)', n_cluster, 'Replicates', 100);
kmeans_idx_schz(kmeans_idx_schz == 1) = 5;
kmeans_idx_schz(kmeans_idx_schz == 3) = 1;
kmeans_idx_schz(kmeans_idx_schz == 5) = 3;

kmeans_idx_schz(kmeans_idx_schz == 1) = 5;
kmeans_idx_schz(kmeans_idx_schz == 2) = 1;
kmeans_idx_schz(kmeans_idx_schz == 5) = 2;

subplot(1,2,2)
scatter3(dm_schz(1, :), dm_schz(2, :), dm_schz(3, :), 40, kmeans_idx_schz, 'filled');
colormap(gca, cmap4)
caxis([0.5, 5.5])

%% brain state distribution
% con_dwell = zeros(n_cluster, 1);
% con_dwell2 = zeros(n_cluster, 1);
% schz_dwell = zeros(n_cluster, 1);
% % match_dict = [1, 4, 2, 3];
% match_dict2 = [3, 1, 2];
% match_dict = [3, 1, 2];
% for i_c = 1 : n_cluster
%     con_dwell(i_c) = sum(kmeans_idx_con == i_c) / n_tp;
%     con_dwell2(i_c) = sum(kmeans_idx_con2 == match_dict2(i_c)) / n_tp;
%     schz_dwell(i_c) = sum(kmeans_idx_schz == match_dict(i_c)) / n_tp;
% end
% 
% figure;
% bar([con_dwell; con_dwell2; schz_dwell])

%% State transition
% trans_con = zeros(n_cluster, n_cluster);
% trans_schz = zeros(n_cluster, n_cluster);
% match_dict = [4, 3, 1, 2];
% for i = 1 : n_tp-1
%     trans_con(kmeans_idx_con(i), kmeans_idx_con(i+1))...
%         = trans_con(kmeans_idx_con(i), kmeans_idx_con(i+1)) + 1;
%     trans_schz(match_dict(kmeans_idx_schz(i)), match_dict(kmeans_idx_schz(i+1)))...
%         = trans_schz(match_dict(kmeans_idx_schz(i)), match_dict(kmeans_idx_schz(i+1))) + 1;
% end
% trans_con = trans_con ./ sum(trans_con, 2);
% trans_schz = trans_schz ./ sum(trans_schz, 2);
% figure;
% subplot(2,1,1)
% imagesc(trans_con)
% caxis([0, 1])
% subplot(2,1,2)
% imagesc(trans_schz)
% caxis([0, 1])

%% Brain state transition for control subjects
n_task = 4;

trans_mat_control = zeros(n_task, n_cluster, n_cluster);
stationary_p_control = zeros(n_task, n_cluster);

for i_task = 1 : n_task
    for j_tp = task_endtime(i_task)+1 : task_endtime(i_task+1)-1
        trans_mat_control(i_task, kmeans_idx_con(j_tp), kmeans_idx_con(j_tp+1))...
            = trans_mat_control(i_task, kmeans_idx_con(j_tp), kmeans_idx_con(j_tp+1)) + 1;
    end
    temp = squeeze(trans_mat_control(i_task, :, :));
    trans_mat_control(i_task, :, :) =  temp ./ sum(temp, 2);
    [temp, ~] = eigs(squeeze(trans_mat_control(i_task, :, :))');
    stationary_p_control(i_task, :) = temp(:, 1) / sum(temp(:, 1));

%     subplot(2,3,i_task)
    figure;
    
    %     heatmap(squeeze(trans_mat(i_task, :, :)), 'Colormap', parula)
    %     caxis([0, 1])
    temp = squeeze(trans_mat_control(i_task,:,:));
    temp(temp<0.1) = 0; % removing small edges
    for i_node = 1 : 4
        for j_node = 1 : 4
            if temp(i_node, j_node)==0 && temp(i_node, j_node)==0
                temp(i_node, j_node) = 0.0001;
            end
        end
    end
    mc = dtmc(temp,'StateNames',stateNames);
    %     h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);
    h = graphplot(mc,'ColorEdges',true);
    
    temp = temp';
    h.LineWidth=log(temp(temp~=0)+1.3)*10;
    h.NodeFontSize = 13;
    h.EdgeFontSize = 10;
    
    temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
    colormap(temp_colormap)
    colorbar('off')
end

% plot stationary distribution
figure;
for i_task = 1 : n_task
    subplot(2, 2, i_task)
    b = bar(stationary_p_control(i_task, :));
    b.FaceColor = 'flat';
    b.CData(1:4, :) = cmap4(1:4, :);
    ylim([0, 0.5])
    title(taskNames(i_task))
end

%% Brain state transition for schz subjects
n_task = 4;

trans_mat_schz = zeros(n_task, n_cluster, n_cluster);
stationary_p_schz = zeros(n_task, n_cluster);

for i_task = 1 : n_task
    for j_tp = task_endtime(i_task)+1 : task_endtime(i_task+1)-1
        trans_mat_schz(i_task, kmeans_idx_schz(j_tp), kmeans_idx_schz(j_tp+1))...
            = trans_mat_schz(i_task, kmeans_idx_schz(j_tp), kmeans_idx_schz(j_tp+1)) + 1;
    end
    temp = squeeze(trans_mat_schz(i_task, :, :));
    trans_mat_schz(i_task, :, :) =  temp ./ sum(temp, 2);
    [temp, ~] = eigs(squeeze(trans_mat_schz(i_task, :, :))');
    stationary_p_schz(i_task, :) = temp(:, 1) / sum(temp(:, 1));

%     subplot(2,3,i_task)
    figure;
    
    %     heatmap(squeeze(trans_mat(i_task, :, :)), 'Colormap', parula)
    %     caxis([0, 1])
    
    temp = squeeze(trans_mat_schz(i_task,:,:));
    temp(temp<0.1) = 0; % removing small edges
    for i_node = 1 : 4
        for j_node = 1 : 4
            if temp(i_node, j_node)==0 && temp(i_node, j_node)==0
                temp(i_node, j_node) = 0.0001;
            end
        end
    end
    mc = dtmc(temp,'StateNames',stateNames);
    %     h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);
    h = graphplot(mc,'ColorEdges',true);
    
    temp = temp';
    h.LineWidth=log(temp(temp~=0)+1.3)*10;
    h.NodeFontSize = 13;
    h.EdgeFontSize = 10;

    temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
    colormap(temp_colormap)
    colorbar('off')
end

% plot stationary distribution
figure;
for i_task = 1 : n_task
    subplot(2, 2, i_task)
    b = bar(stationary_p_schz(i_task, :));
    b.FaceColor = 'flat';
    b.CData(1:4, :) = cmap4(1:4, :);
    ylim([0, 0.5])
    title(taskNames(i_task))
end

%% entropy transition calculation
% get rs_kmeans_idx
% [~, I] = pdist2(dm(1:4, :)', dm_all(1:4, test_range)', 'euclidean','Smallest', 1);
% rs_kmeans_idx = kmeans_idx(I);
% kmeans_idx_all = [kmeans_idx; rs_kmeans_idx];

% control
task_entropy_control = zeros(n_task, n_cluster);
% figure;
hold on
for i_task = 1 : n_task
    temp_trans_mat = squeeze(trans_mat_control(i_task, :, :));
    
    for j_state = 1 : 4
        temp_pmf = temp_trans_mat(j_state,:);
        temp_pmf(temp_pmf==0) = [];
        task_entropy_control(i_task, j_state) = -sum(temp_pmf.*log(temp_pmf));
    end
    
%     task_entropy_control(i_task, stationary_p_control(i_task, :) < 0.2) = 0;
    scatter([1:4]-offset, task_entropy_control(i_task, :), 150, cmap_ucla(i_task+1,:), 'filled')
    
    ylim([0.5, 1.4])
%     set(gca,'xtick',[])
%     set(gca,'ytick',[])
%     set(gca,'visible','off')
end

% schz
task_entropy_schz = zeros(n_task, n_cluster);

for i_task = 1 : n_task
    temp_trans_mat = squeeze(trans_mat_schz(i_task, :, :));
    
    for j_state = 1 : 4
        temp_pmf = temp_trans_mat(j_state,:);
        temp_pmf(temp_pmf==0) = [];
        task_entropy_schz(i_task, j_state) = -sum(temp_pmf.*log(temp_pmf));
    end
    
%     task_entropy_schz(i_task, stationary_p_schz(i_task, :) < 0.2) = 0;
        
    scatter([1:4]+offset, task_entropy_schz(i_task, :), 150, cmap_ucla(i_task+1,:), 'd', 'filled')
    
    ylim([0.5, 1.4])
end
hold off

% average
figure;
scatter(1:4, mean(task_entropy_control), 150, 'filled')
hold on
scatter(1:4, mean(task_entropy_schz), 150, 'd', 'filled')
hold off

% average using only high-prob task
figure;
scatter(1:4, mean(task_entropy_control), 150, 'filled')
hold on
scatter(1:4, mean(task_entropy_schz), 150, 'd', 'filled')
hold off

%% Calculate Participation Coefficient
% create temporal label
task_idx = [[1,242]; [243, 510]; [511, 801]; [802, 1009]];
temporal_label = zeros(1, n_tp);
for i_task = 1 : size(task_idx, 1)
    temporal_label(task_idx(i_task, 1):task_idx(i_task, 2)) = i_task;
end

% for control subject
pc_global_con = computeDynamicNetworkMeasureDynamicCommunity(ucla_task(:, tp_idx, ...
    sub_idx_control), 15, temporal_label);

% exmaine box plot
nonzero_pc_idx = pc_global_con~=0;
figure;
boxplot(pc_global_con(nonzero_pc_idx), kmeans_idx_con(nonzero_pc_idx))

% for schz subject
pc_global_schz = computeDynamicNetworkMeasure(ucla_task(:, tp_idx, ...
    sub_idx_schz), map268, 15, temporal_label);

% exmaine box plot
figure;
boxplot(pc_global_schz(nonzero_pc_idx), kmeans_idx_schz(nonzero_pc_idx))

% color embedding by participation coefficient
figure;
scatter(dm_con(1, nonzero_pc_idx), dm_con(2, nonzero_pc_idx),...
    40, pc_global_con(nonzero_pc_idx), 'filled');
temp_corr = corr(dm_con(:, nonzero_pc_idx)', pc_global_con(nonzero_pc_idx)')
xlim([-0.05, 0.04])
colormap(jet)
caxis([0.42, 0.5])

figure;
scatter3(dm_schz(1, nonzero_pc_idx), -dm_schz(2, nonzero_pc_idx),...
    dm_schz(3, nonzero_pc_idx), 40, pc_global_schz(nonzero_pc_idx), 'filled');
temp_corr = corr(dm_schz(:, nonzero_pc_idx)', pc_global_schz(nonzero_pc_idx)')
colormap(jet)
caxis([0.822, 0.842])

%% plot two box plots together
% by 3 cluster
positions = [1 2 3 2.25 3.25 1.25];
colors = 'rb';
figure;
boxplot([pc_global_con(nonzero_pc_idx), pc_global_schz(nonzero_pc_idx)], ...
[kmeans_idx_con(nonzero_pc_idx); kmeans_idx_schz(nonzero_pc_idx)+3],...
'positions', positions, 'Colors', colors)

% by 4 cluster
positions = [1 2 3 4 1.25 2.25 3.25 4.25];
% positions = [3 1 4 2 4.25 1.25 3.25 2.25];
colors = 'rb';
figure;
boxplot([pc_global_con(nonzero_pc_idx), pc_global_schz(nonzero_pc_idx)], ...
[kmeans_idx_con(nonzero_pc_idx); kmeans_idx_schz(nonzero_pc_idx)+4],...
'positions', positions, 'Colors', colors)

% mean difference of two clusters
mean_diff = zeros(4, 1);
c1 = pc_global_con(nonzero_pc_idx);
c2 = pc_global_schz(nonzero_pc_idx);
match_idx = [3, 2, 1, 4];
for i = 1 : 4
    mean_diff(i) = mean(c2(kmeans_idx_schz(nonzero_pc_idx)==match_idx(i)))-...
        mean(c1(kmeans_idx_con(nonzero_pc_idx)==i));
end
figure;
bar(mean_diff([2,4,1,3]))

% cohen's D of two clusters
cohend = zeros(4, 1);
c1 = pc_global_con(nonzero_pc_idx);
c2 = pc_global_schz(nonzero_pc_idx);
match_idx = [1, 2, 3, 4];
for i = 1 : 4
    c1_temp = c1(kmeans_idx_con(nonzero_pc_idx)==match_idx(i));
    c2_temp = c2(kmeans_idx_schz(nonzero_pc_idx)==match_idx(i));
    cohend(i) = computeCohen_d(c2_temp, c1_temp, 'independent');
end
figure;
bar(cohend([1,2,3,4]))
    
    
% by task
% create temporal label
task_idx = [[1,242]; [243, 510]; [511, 801]; [802, 1009]];
temporal_label = zeros(1, n_tp);
for i_task = 1 : size(task_idx, 1)
    temporal_label(task_idx(i_task, 1):task_idx(i_task, 2)) = i_task;
end

positions = [1 2 3 4 1.25 2.25 3.25 4.25];
colors = 'rb';
figure;
boxplot([pc_global_con(nonzero_pc_idx), pc_global_schz(nonzero_pc_idx)], ...
[temporal_label(nonzero_pc_idx), temporal_label(nonzero_pc_idx)+4],...
'positions', positions, 'Colors', colors)

positions = [1:11, 1.25:11.25];
colors = 'rb';
figure;
boxplot([pc_global_con(nonzero_pc_idx), pc_global_schz(nonzero_pc_idx)], ...
[all_labels(nonzero_pc_idx), all_labels(nonzero_pc_idx)+11],...
'positions', positions, 'Colors', colors)


%% run t-test to test if in control high cog > low cog
c1 = pc_global_con(nonzero_pc_idx);
c1 = c1(kmeans_idx_con(nonzero_pc_idx)==1);
c2 = pc_global_con(nonzero_pc_idx);
c2 = c2(kmeans_idx_con(nonzero_pc_idx)==2);
[h, p] = ttest2(c1, c2, 'Tail', 'left')

c3 = pc_global_schz(nonzero_pc_idx);
c3 = c3(kmeans_idx_schz(nonzero_pc_idx)==1);
c4 = pc_global_schz(nonzero_pc_idx);
c4 = c4(kmeans_idx_schz(nonzero_pc_idx)==2);
[h, p] = ttest2(c3, c4, 'Tail', 'right')

c_con = pc_global_con(nonzero_pc_idx);
c_con = c_con(kmeans_idx_con(nonzero_pc_idx)==3);
c_schz = pc_global_schz(nonzero_pc_idx);
c_schz = c_schz(kmeans_idx_schz(nonzero_pc_idx)==4);
[h, p] = ttest2(c_con, c_schz, 'Tail', 'left')

%% cluster by PCA reduced dimension
% control
n_dim = 20;
n_sub_control = numel(sub_idx_control);
embed = zeros(n_tp, n_sub_control*n_dim);
for i_sub = 1 : n_sub_control 
    data_ind = ucla_task(:, :, sub_idx_control(i_sub));
    [~, score, ~, ~, explained, ~] = pca(data_ind', 'NumComponents', n_dim);
    embed(:, (i_sub-1)*n_dim+1:(i_sub)*n_dim) = score;
%     sum(explained(1:n_dim))/100
end
[~, score, ~,~,explained,~] = pca(embed, 'NumComponents', n_dim);
% sum(explained(1:n_dim))/100

rng(665)
k = 3;
kmeans_idx_pca_con = kmeans(score(:, :), k, 'Replicates', 100);

figure;
scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, kmeans_idx_pca_con, 'filled');
colormap(cmap12)
caxis([vmin, vmax])

% schz
n_dim = 20;
n_sub_schz = numel(sub_idx_schz);
embed = zeros(n_tp, n_sub_schz*n_dim);
for i_sub = 1 : n_sub_schz 
    data_ind = ucla_task(:, :, sub_idx_schz(i_sub));
    [~, score, ~, ~, explained, ~] = pca(data_ind', 'NumComponents', n_dim);
    embed(:, (i_sub-1)*n_dim+1:(i_sub)*n_dim) = score;
%     sum(explained(1:n_dim))/100
end
[~, score, ~,~,explained,~] = pca(embed, 'NumComponents', n_dim);
% sum(explained(1:n_dim))/100

rng(665)
k = 3;
kmeans_idx_pca_schz = kmeans(score(:, :), k, 'Replicates', 100);

figure;
scatter3(dm_schz(1, :), dm_schz(2, :), dm_schz(3, :), 40, kmeans_idx_pca_schz, 'filled');
colormap(cmap12)
caxis([vmin, vmax])

% boxplots by cluster
positions = [1 2 3 1.25 2.25 3.25];
% colors = ['r', 'r', 'r', 'b', 'b', 'b'];
colors = 'rb';
figure;
boxplot([pc_global_con(nonzero_pc_idx), pc_global_schz(nonzero_pc_idx)], ...
[kmeans_idx_pca_con(nonzero_pc_idx); kmeans_idx_pca_schz(nonzero_pc_idx)+3],...
'positions', positions, 'Colors', colors)

% t-test
c_con = pc_global_con(nonzero_pc_idx);
c_con = c_con(kmeans_idx_pca_con(nonzero_pc_idx)==1);
c_schz = pc_global_schz(nonzero_pc_idx);
c_schz = c_schz(kmeans_idx_pca_schz(nonzero_pc_idx)==2);
[h, p] = ttest2(c_con, c_schz, 'Tail', 'left')

%% PC difference plot
dm_schz_new = dm_schz;
dm_schz_new(2,:) = -dm_schz_new(2,:);

figure;
scatter3(dm_schz_new(1, :), dm_schz_new(2, :), dm_schz_new(3, :), 40, 'red', 'filled');
hold on
scatter3(dm_con(1, :), dm_con(2, :), dm_con(3, :), 40, 'blue', 'filled');

[D, I] = pdist2(dm_con(1:3, :)', dm_schz_new(1:3, :)', 'euclidean', 'Smallest', 1);

diff = pc_global_schz - pc_global_con;
nonzero_pc_idx_diff = abs(diff)<0.8;

figure;
scatter3(dm_schz(1, nonzero_pc_idx_diff), dm_schz(2, nonzero_pc_idx_diff),...
    dm_schz(3, nonzero_pc_idx_diff), 40, diff(nonzero_pc_idx_diff), 'filled');
colormap(jet)
% caxis([0, 0.015])
