% This script validates the rs dynamics of male-female cohort in HCP
% dataset when males are extended onto the embedding generated from all the
% subjects instead of only the male subjects

%% add path
root_path = '/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM';
addpath(fullfile(root_path, 'utils'))
addpath(fullfile(root_path, 'src'))
addpath(fullfile(root_path, 'utils/BrainSync'))
addpath('/Users/siyuangao/Working_Space/fmri/bctnet/BCT/2017_01_15_BCT/')

%% load data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map259.mat')
% 1-female 2-male
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/all_sex390.mat')

% choose less subjects for quicker computation
data = data(:, :, 1:390);
all_sex = all_sex(1:390);
% all_sex = all_sex(randperm(100));

% % zscore the region
% for i_sub = 1 : size(data,3)
%     data(:, :, i_sub) = zscore(data(:, :, i_sub), 0, 2);
% end

task_endtime = [0,772,1086,1554,2084,2594,3020,4176];
% task_length = [772,314,468,530,510,426]
test_range = 3021:4176;
train_range = 1:3020;

test_data = data(:, test_range, :);
n_tp_test = size(test_data, 2); % length of testing data

train_data = data(:, train_range, :); % task data to generate manifold
n_sub_all = size(train_data, 3); % number of subjects
n_tp_train = size(train_data, 2); % length of training data
n_region = size(train_data, 1); % number of regions
n_tp_all = n_tp_test+n_tp_train; % total length of time

cohorts_name = {'female', 'male'};
states_name = {'high cog', 'transition', 'fixation', 'low cog'};

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
n_cohorts = 2;

% embed is the n_t*(n_sub*n_d) first round flatten matrix, dm is the final embedding
[dm_all, ~, embed_all, lambda1_all, lambda2_all, sigma1_all, ...
    sigma2_all] = calc2sDM(train_data, n_dim, configAffParams1,...
    configAffParams2, configDiffParams1, configDiffParams2);

embed_all = reshape(embed_all, [], n_sub_all, n_dim);
figure;
subplot(2, 1, 1)
c = true_label_all(train_range);
scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), 20, c, 'filled');
colormap(cmap10)
caxis([-0.5, 9.5])

% kmeans part
rng(665)
n_cluster = 4;
kmeans_idx = kmeans(dm_all(1:4, :)', n_cluster, 'Replicates', 100);
kmeans_map = [4,2,1,3];
for i = 1 : n_cluster
    kmeans_idx(kmeans_idx==i)=kmeans_map(i)+n_cluster; 
end
kmeans_idx = kmeans_idx - n_cluster;
subplot(2, 1, 2)
scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), 20, kmeans_idx, 'filled');
colormap(gca, cmap4)
caxis([0.5, 5.5])

%% generate extension embedding and calculate dynamics (distributon,entropy)
all_trans_mat_rs = cell(n_cohorts, 1);
all_stationary_p_rs = cell(n_cohorts, 1);

k_ext = 500;
configAffParams3 = configAffParams1;
configAffParams3.kNN = k_ext;

for i_idx = 1 : n_cohorts
    n_sub = sum(all_sex==i_idx);
    all_trans_mat_rs_temp = zeros(n_sub, n_cluster, n_cluster);
    all_stationary_p_rs_temp = zeros(n_sub, n_cluster);
    
    all_task_temp = train_data(:, :, all_sex==i_idx);
    all_rest_temp = test_data(:, :, all_sex==i_idx);
    
    train_range = 1 : n_tp_train;
    test_range = n_tp_train+1 : n_tp_all;

    sigma1_iidx = sigma1_all;
    embed_iidx = embed_all;
    lambda1_iidx = lambda1_all;
    figure;
    tic
    for ref_sub = 1 : n_sub
        disp(ref_sub)
        
        data_ind = zeros(n_region, n_tp_all);
        data_ind(:, train_range) = all_task_temp(:, :, ref_sub);
        data_ind(:, test_range) = all_rest_temp(:, :, ref_sub);
        
        configAffParams3_temp = configAffParams3;
        configAffParams3_temp.sig = sigma1_iidx(i);
        K = calcAffinityMat(data_ind, configAffParams3_temp);
        K = K(test_range, train_range);
        K = K./sum(K, 2);
        
        psi2 = K*dm_all'./lambda1_iidx(:, i)';
        
        dm_all_temp = zeros(n_dim, n_tp_all);
        dm_all_temp(:, train_range) = dm_all;
        dm_all_temp(:, test_range) = psi2';
        
        c = true_label_all(1:n_tp_all);
        scatter3(dm_all_temp(1, :), dm_all_temp(2, :),dm_all_temp(3, :),...
            20, c, 'filled');
        colormap(cmap10)
        caxis([-0.5, 9.5])
        drawnow
        % get transition matrix and stationary distribution
        % get rs_kmeans_idx
        [~, I] = pdist2(dm_all(1:3, :)', ...
            dm_all_temp(1:3, test_range)', 'euclidean','Smallest', 1);
        rs_kmeans_idx = kmeans_idx(I);

        trans_mat_rs_temp = zeros(n_cluster, n_cluster);
        for i_tp = 1 : numel(test_range)-1
            %     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
            c1 = rs_kmeans_idx(i_tp);
            c2 = rs_kmeans_idx(i_tp+1);
            trans_mat_rs_temp(c1, c2) = trans_mat_rs_temp(c1, c2) + 1;
            %     end
        end
        trans_mat_rs_temp = trans_mat_rs_temp./ sum(trans_mat_rs_temp, 2);
        trans_mat_rs_temp(isnan(trans_mat_rs_temp)) = 0;
        [temp, ~] = eigs(trans_mat_rs_temp');
        stationary_p_rs_temp = temp(:, 1) / sum(temp(:, 1));

        all_trans_mat_rs_temp(ref_sub, :, :) = trans_mat_rs_temp;
        all_stationary_p_rs_temp(ref_sub, :) = stationary_p_rs_temp;
        
    end
    toc
    all_trans_mat_rs{i_idx} = all_trans_mat_rs_temp;
    all_stationary_p_rs{i_idx} = all_stationary_p_rs_temp;
end
figure;
subplot(1,2,1)
boxchart(all_stationary_p_rs{1})
ylim([0, 0.6])
title('female')
subplot(1,2,2)
boxchart(all_stationary_p_rs{2})
ylim([0, 0.6])
title('male')

p_all_dist = zeros(n_cluster, 1);
real_t_dist = zeros(n_cluster, 1);
for i_c = 1 : n_cluster
    [~, p_all_dist(i_c), ~, stats] = ttest2(all_stationary_p_rs{1}(:, i_c), ...
        all_stationary_p_rs{2}(:, i_c));
    real_t_dist(i_c) = stats.tstat;
end

% entropy
rs_entropy_all = cell(n_cohorts, 1);
for i_idx = 1 : n_cohorts
    n_sub = sum(all_sex==i_idx);
    rs_entropy_temp = zeros(n_sub, n_cluster);
    for i_sub = 1 : n_sub
        for j_state = 1 : n_cluster
            temp_pmf = all_trans_mat_rs{i_idx}(i_sub, j_state,:);
            temp_pmf(temp_pmf==0) = [];
            rs_entropy_temp (i_sub, j_state) = -sum(temp_pmf.*log(temp_pmf));
        end
    end
    rs_entropy_all{i_idx} = rs_entropy_temp;
end

figure;
subplot(1,2,1)
boxchart(rs_entropy_all{1})
ylim([0, 1.4])
title('female')
subplot(1,2,2)
boxchart(rs_entropy_all{2})
ylim([0, 1.4])
title('male')

p_all_entropy = zeros(n_cluster, 1);
real_t_entropy = zeros(n_cluster, 1);
for i_c = 1 : n_cluster
    [~, p_all_entropy(i_c), ~, stats] = ttest2(rs_entropy_all{1}(:, i_c), ...
        rs_entropy_all{2}(:, i_c));
    real_t_entropy(i_c) = stats.tstat;
end

disp(p_all_entropy)

%% null distribution
rng(665)
n_perm = 1000;
n_sub1 = sum(all_sex==1);
n_sub2 = sum(all_sex==2);

all_dist = [all_stationary_p_rs{1};all_stationary_p_rs{2}];
all_entropy = [rs_entropy_all{1};rs_entropy_all{2}];

t_dist = zeros(n_perm, n_cluster);
t_entropy = zeros(n_perm, n_cluster);
for i_perm = 1 : n_perm
    perm_idx = randperm(390); % generate permuted sex label
    all_dist_rand = all_dist(perm_idx, :);
    all_entropy_rand = all_entropy(perm_idx, :);
    
    for i_c = 1 : n_cluster
        [~, ~, ~, stats] = ttest2(all_dist_rand(1:n_sub1, i_c), ...
            all_dist_rand(n_sub1+1:end, i_c));
        t_dist(i_perm, i_c) = stats.tstat;
        [~, ~, ~, stats] = ttest2(all_entropy_rand(1:n_sub1, i_c), ...
            all_entropy_rand(n_sub1+1:end, i_c));
        t_entropy(i_perm, i_c) = stats.tstat;
    end
end

% comparison figure
figure;
for i_c = 1 : n_cluster
    subplot(1, 4, i_c)
    histogram(t_dist(:, i_c))
    xline(real_t_dist(i_c))
    xlim([-6, 6])
    p_val = min(sum(real_t_dist(i_c)>t_dist(:, i_c))/n_perm,...
        sum(real_t_dist(i_c)<t_dist(:, i_c))/n_perm);
    title(sprintf('state: %s, p: %.3f', states_name{i_c}, p_val))
end

figure;
for i_c = 1 : n_cluster
    subplot(1, 4, i_c)
    histogram(t_entropy(:, i_c))
    xline(real_t_entropy(i_c))
    xlim([-6, 6])
    p_val = min(sum(real_t_entropy(i_c)>t_dist(:, i_c))/n_perm,...
        sum(real_t_entropy(i_c)<t_dist(:, i_c))/n_perm);
    title(sprintf('state: %s, p: %.3f', states_name{i_c}, p_val))
end
%% Brain state transition
% get rs_kmeans_idx
[~, I] = pdist2(dm(1:4, :)', dm_all(1:4, test_range)', 'euclidean','Smallest', 1);
rs_kmeans_idx = kmeans_idx(I);
kmeans_idx_all = [kmeans_idx; rs_kmeans_idx];

n_task = 6;

trans_mat = zeros(n_task, n_cluster, n_cluster);
stationary_p = zeros(n_task, n_cluster);
stateNames = ["High cog" "Transition" "Fixation" "Low cog"];
for i_task = 1 : n_task
    for j_tp = task_endtime(i_task)+1 : task_endtime(i_task+1)-1
        trans_mat(i_task, kmeans_idx_all(j_tp), kmeans_idx_all(j_tp+1))...
            = trans_mat(i_task, kmeans_idx_all(j_tp), kmeans_idx_all(j_tp+1)) + 1;
    end
    temp = squeeze(trans_mat(i_task, :, :));
    trans_mat(i_task, :, :) =  temp ./ sum(temp, 2);
    [temp, ~] = eigs(squeeze(trans_mat(i_task, :, :))');
    stationary_p(i_task, :) = temp(:, 1) / sum(temp(:, 1));

%     subplot(2,3,i_task)
    figure;
    
    %     heatmap(squeeze(trans_mat(i_task, :, :)), 'Colormap', parula)
    %     caxis([0, 1])
    
    temp = squeeze(trans_mat(i_task,:,:));
    temp(temp<0.03) = 0; % removing small edges
    if temp(2,3)==0 && temp(3,2)==0
        temp(2,3) = 0.0001;
    end
    mc = dtmc(temp,'StateNames',stateNames);
    %     h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);
    h = graphplot(mc,'ColorEdges',true);
    
    %     temp = temp';
    %     h.LineWidth=log(temp(temp~=0)+1.3)*10;
    h.LineWidth = 2;
    %     h.NodeFontSize = 0;
    %     h.EdgeFontSize = 10;
    %     h.NodeLabelColor = [1, 1, 1];
    h.NodeLabel = {};
    
%     Cdata = colormap(flipud(gray)); % node color
    temp_colormap = [[0, 0, 0];flipud(autumn)]; % edge color
    colormap(temp_colormap)
    %     colormap(flipud(autumn))
    %     colormap(jet)
    
    %     Cdata = colormap(flipud(autumn));
%     colormap(flipud(gray))
%     caxis([-0.5, 1])
    
%     temp_p = stationary_p(i_task, :);
%     c_idx = floor(temp_p ./ 0.6 .* 256);
%     h.NodeColor = Cdata(c_idx, :);
    colorbar('off')
    set(gca,'color','k');
end

% WM blocks transition
% 0back
temporal_idx = find(true_label_all==2);
trans_mat_0back = zeros(n_cluster, n_cluster);
for i_tp = 1 : numel(temporal_idx)-1
%     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = kmeans_idx_all(temporal_idx(i_tp));
        c2 = kmeans_idx_all(temporal_idx(i_tp+1));
        trans_mat_0back(c1, c2) = trans_mat_0back(c1, c2) + 1;
%     end
end
trans_mat_0back = trans_mat_0back./ sum(trans_mat_0back, 2);
figure;
temp = trans_mat_0back;
temp(temp<0.01) = 0; % removing small edges
if temp(2,3)==0 && temp(3,2)==0
    temp(2,3) = 0.0001;
end
mc = dtmc(temp,'StateNames',stateNames);
h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);

temp = temp';
h.LineWidth=log(temp(temp~=0)+1.3)*10;
h.NodeFontSize = 13;
h.EdgeFontSize = 10;

Cdata = colormap(flipud(gray)); % node color
temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
colormap(temp_colormap)

[temp, ~] = eigs(trans_mat_0back');
stationary_p_0back = temp(:, 1) / sum(temp(:, 1));
c_idx = floor(stationary_p_0back ./ 0.6 .* 256);
h.NodeColor = Cdata(c_idx, :);
colorbar('off')

% 2back
temporal_idx = find(true_label_all==3);
trans_mat_0back = zeros(n_cluster, n_cluster);
for i_tp = 1 : numel(temporal_idx)-1
%     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = kmeans_idx_all(temporal_idx(i_tp));
        c2 = kmeans_idx_all(temporal_idx(i_tp+1));
        trans_mat_0back(c1, c2) = trans_mat_0back(c1, c2) + 1;
%     end
end
trans_mat_0back = trans_mat_0back./ sum(trans_mat_0back, 2);
figure;
temp = trans_mat_0back;
temp(temp<0.01) = 0; % removing small edges
if temp(2,3)==0 && temp(3,2)==0
    temp(2,3) = 0.0001;
end
mc = dtmc(temp,'StateNames',stateNames);
h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);

temp = temp';
h.LineWidth=log(temp(temp~=0)+1.3)*10;
h.NodeFontSize = 13;
h.EdgeFontSize = 10;

Cdata = colormap(flipud(gray)); % node color
temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
colormap(temp_colormap)

[temp, ~] = eigs(trans_mat_0back');
stationary_p_2back = temp(:, 1) / sum(temp(:, 1));
c_idx = floor(stationary_p_2back ./ 0.6 .* 256);
h.NodeColor = Cdata(c_idx, :);
colorbar('off')

% plot stationary distribution
figure;
subplot(3,3,1)
b = bar(stationary_p_0back);
b.FaceColor = 'flat';
b.CData(1:4, :) = cmap4(1:4, :);
ylim([0, 0.6])
subplot(3,3,2)
b = bar(stationary_p_2back);
b.FaceColor = 'flat';
b.CData(1:4, :) = cmap4(1:4, :);
ylim([0, 0.6])
for i_task = 1 : n_task
    subplot(3, 3, i_task+2)
    b = bar(stationary_p(i_task, :));
    b.FaceColor = 'flat';
    b.CData(1:4, :) = cmap4(1:4, :);
    ylim([0, 0.6])
end

% calculate skewness
% skew = zeros(n_task+2, 1);
% temp_mean = (1:4) * stationary_p_0back;
% temp_m = ((1:4) - temp_mean).^3 * stationary_p_0back;
% temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p_0back'));
% skew(1) = temp_m / temp_std^3;
% temp_mean = (1:4) * stationary_p_2back;
% temp_m = ((1:4) - temp_mean).^3 * stationary_p_2back;
% temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p_2back'));
% skew(2) = temp_m / temp_std^3;
% for i_task = 1 : n_task
%     temp_mean = (1:4) * stationary_p(i_task, :)';
%     temp_m = ((1:4) - temp_mean).^3 * stationary_p(i_task, :)';
%     temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p(i_task, :)));
%     skew(i_task+2) = temp_m / temp_std^3;
% end

%% dState1 vs dState2
stationary_all = [stationary_p_0back';stationary_p_2back';stationary_p];

x = stationary_all(:, [1, 4]);
figure;
scatter(x(:, 1), x(:, 2), 'filled', 'r');

% chi-square goodness of fit test
cs_all = zeros(8, 1);

% 0back
state_num_tp = zeros(4, 1);
temp_range = find(true_label_all==2);
for j_state = 1 : 4
    state_num_tp(j_state) = sum(kmeans_idx_all(temp_range) == j_state);
end
expected = [numel(temp_range)*0.25, numel(temp_range)*0.25, ...
    numel(temp_range)*0.25, numel(temp_range)*0.25];
[h,p,st] = chi2gof(1:4,'Ctrs',1:4,'Frequency',state_num_tp, ...
    'Expected',expected);
cs_all(1) = st.chi2stat;

% 2back
state_num_tp = zeros(4, 1);
temp_range = find(true_label_all==3);
for j_state = 1 : 4
    state_num_tp(j_state) = sum(kmeans_idx_all(temp_range) == j_state);
end
expected = [numel(temp_range)*0.25, numel(temp_range)*0.25, ...
    numel(temp_range)*0.25, numel(temp_range)*0.25];
[h,p,st] = chi2gof(1:4,'Ctrs',1:4,'Frequency',state_num_tp, ...
    'Expected',expected);
cs_all(2) = st.chi2stat;

% other tasks
for i_task = 1 : n_task
    state_num_tp = zeros(4, 1);
    temp_range = task_endtime(i_task)+1 : task_endtime(i_task+1)-1;
    for j_state = 1 : 4
        state_num_tp(j_state) = sum(kmeans_idx_all(temp_range) == j_state);
    end
    expected = [numel(temp_range)*0.25, numel(temp_range)*0.25, ...
        numel(temp_range)*0.25, numel(temp_range)*0.25];
    [h,p,st] = chi2gof(1:4,'Ctrs',1:4,'Frequency',state_num_tp, ...
                        'Expected',expected);
    cs_all(i_task+2) = st.chi2stat;
end

%% RS transition
temporal_idx = find(true_label_all(1:n_tp_all)==9);
trans_mat_rs = zeros(n_cluster, n_cluster);
for i_tp = 1 : numel(temporal_idx)-1
%     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = kmeans_idx_all(temporal_idx(i_tp));
        c2 = kmeans_idx_all(temporal_idx(i_tp+1));
        trans_mat_rs(c1, c2) = trans_mat_rs(c1, c2) + 1;
%     end
end
trans_mat_rs = trans_mat_rs./ sum(trans_mat_rs, 2);
[temp, ~] = eigs(trans_mat_rs');
stationary_p_rs= temp(:, 1) / sum(temp(:, 1));
figure;
b = bar(stationary_p_rs);
b.FaceColor = 'flat';
b.CData(1:4, :) = cmap4(1:4, :);
ylim([0, 0.6])

temp_mean = (1:4) * stationary_p_rs;
temp_m = ((1:4) - temp_mean).^3 * stationary_p_rs;
temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p_rs'));
skew_rs = temp_m / temp_std^3;

%% entropy transition calculation
task_entropy = zeros(7, 4);
figure;
hold on
% colors = [[0,0.23,1];[1,0.75,0];[0.25, 1, 0.75];[0.6, 0, 0.6];[0.65,0.16,0.16];[0.86,0.44,0.58]];
for i_task = 1 : 6
    if i_task <=6
        temp_trans_mat = squeeze(trans_mat(i_task, :, :));
    else
        temp_trans_mat = trans_mat_rs;
    end
    for j_state = 1 : 4
        temp_pmf = temp_trans_mat(j_state,:);
        temp_pmf(temp_pmf==0) = [];
        task_entropy(i_task, j_state) = -sum(temp_pmf.*log(temp_pmf));
    end
    if i_task <= 6
        scatter(1:4, task_entropy(i_task, :), 150, cmap10(i_task+3,:), 'filled')
    else
        scatter(1:4, task_entropy(i_task, :), 'r')
    end
    ylim([0, 1.4])
end
bh = boxplot(rs_entropy, 'Colors', cmap4(5, :))
set(bh, 'LineWidth', 2)
ylim([0, 1.4])
hold off

%% Distance of RS to the other task
js_dist = zeros(9, 9);
stationary_all = [stationary_p_0back';stationary_p_2back';stationary_p; ...
    stationary_p_rs'];
for i_task = 1 : 9
    temp_idx = logical(ones(9, 1));
    temp_idx(i_task) = false;
    js_dist(i_task, temp_idx) = JSDiv(stationary_all(temp_idx, :), ...
        stationary_all(i_task, :));
end
adj = 1 - js_dist;
% adj(adj<mean(adj)) = 0;
adj(1:8, :) = 0;
adj = adj*100;
G = graph(adj, {'0back', '2back', 'WM', 'EMO', 'GAM', 'MOT', 'SOC', ...
    'REL', 'RS'}, 'lower', 'omitselfloops');

% figure; plot(G, 'Layout','force', 'WeightEffect', 'inverse')
figure; plot(G, 'Layout','force', 'EdgeLabel', G.Edges.Weight)

%% color resting-state by particiation coefficient
% align subjects
ref_sub = 1;
rs_aligned = zeros(n_region, n_tp_test, n_sub);
for i = 1 : n_sub
    disp(i)
    rs_aligned(:, :, i) = test_data(:, :, i);
    if i ~= ref_sub
        [Y2, ~] = brainSync(test_data(:, :, ref_sub)', rs_aligned(:, :, i)');
        rs_aligned(:, :, i) = Y2';
    end
end

rs_pc = computeDynamicNetworkMeasure(rs_aligned, map259, 15);
rs_pc(rs_pc==0) = mean(rs_pc(rs_pc~=0));
figure;
scatter3(dm_all(1, test_range), dm_all(2, test_range), dm_all(4, test_range), 20, rs_pc, 'filled');
corr(dm_all(:, test_range)', rs_pc')
colormap(jet)

figure;
scatter(dm_all(1, test_range), rs_pc)

