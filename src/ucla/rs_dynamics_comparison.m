% choose less subjects for quicker computation
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/idx199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/rest199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_labels.mat')
load('/Users/siyuangao/Working_Space/fmri/data/colormaps/cmap12.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap6.mat')

data = data_label_generation(1);

% zscore RS by region
for i_sub = 1 : 199
    all_signal(:, :, i_sub) = zscore(all_signal(:, :, i_sub), 0, 1);
end


sub_idx_control = idx_control(1:77);
sub_idx_schz = idx_schz(1:44);

all_task_con = data(:, :, idx_control);
all_rest_con = all_signal(:, :, idx_control);

all_task_schz = data(:, :, idx_schz);
all_rest_schz = all_signal(:, :, idx_schz);

n_tp_task = size(all_task_con, 2);
n_tp_rest = size(all_rest_con, 2);
n_tp_all = n_tp_task + n_tp_rest;
n_sub_con = size(all_task_con, 3);
n_sub_schz = size(all_task_schz, 3);
n_region = size(all_task_con, 1);

% algorithm unrelated unconfig
rng(665)
vmin = -0.5;
vmax = 11.5;

train_range = 1 : n_tp_task;
test_range = n_tp_task+1 : n_tp_all;

%% control
% generate training embedding
configAffParams1.kNN = 200;
configAffParams2.kNN = 200;
[dm_con, ~, embed_con, lambda1_con, lambda2_con, sigma1_con, sigma2_con] = ...
    calc2sDM(all_task_con, n_dim, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

% generate extension embedding
all_trans_mat_rs_con = zeros(n_sub_con, n_cluster, n_cluster);
all_stationary_p_rs_con = zeros(n_sub_con, n_cluster);
for ref_sub = 1 : n_sub_con
    
    k_ext = 100;
    
    psi1 = zeros(n_tp_rest, n_dim*n_sub_con);
    configAffParams3 = configAffParams1;
    configAffParams3.kNN = k_ext;
    all_rest_sync = zeros(n_region, n_tp_rest, n_sub_con);
    
    for i = 1 : n_sub_con
        disp(i)
        data_ind = zeros(n_region, n_tp_all);
        data_ind(:, train_range) = all_task_con(:, :, i);
        data_ind(:, test_range) = all_rest_con(:, :, i);
        if i ~= ref_sub
            [Y2, R] = brainSync(all_rest_con(:, :, ref_sub)', data_ind(:, test_range)');
            data_ind(:, test_range) = Y2';
        end
        all_rest_sync(:, :, i) = data_ind(:, test_range);
        configAffParams3.sig = sigma1_con(i);
        [K] = calcAffinityMat(data_ind, configAffParams3);
        K = K(test_range, train_range);
        K = K./sum(K, 2);
        psi1(:, (i-1)*n_dim+1:i*n_dim) = K*embed_con(:, (i-1)*n_dim+1:i*n_dim)./lambda1_con(:, i)';
    end
    
    % second round embedding
    embed_all = zeros(n_dim*n_sub_con, n_tp_all);
    embed_all(:, train_range) = embed_con';
    embed_all(:, test_range) = psi1';
    configAffParams3.sig = sigma2_con;
    [K, ~] = calcAffinityMat(embed_all, configAffParams3);
    K = K(test_range, train_range);
    K = K./sum(K, 2);
    psi2 = K*dm_con'./lambda2_con';
    
    dm_con_all = zeros(n_dim, n_tp_all);
    dm_con_all(:, train_range) = dm_con;
    dm_con_all(:, test_range) = psi2';
    
    % get transition matrix and stationary distribution
    % get rs_kmeans_idx
    [~, I] = pdist2(dm_con(1:3, :)', dm_con_all(1:3, test_range)', 'euclidean','Smallest', 1);
    rs_kmeans_idx = kmeans_idx_con(I);
    
    trans_mat_rs_con = zeros(n_cluster, n_cluster);
    for i_tp = 1 : numel(test_range)-1
        %     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = rs_kmeans_idx(i_tp);
        c2 = rs_kmeans_idx(i_tp+1);
        trans_mat_rs_con(c1, c2) = trans_mat_rs_con(c1, c2) + 1;
        %     end
    end
    trans_mat_rs_con = trans_mat_rs_con./ sum(trans_mat_rs_con, 2);
    [temp, ~] = eigs(trans_mat_rs_con');
    stationary_p_rs_con = temp(:, 1) / sum(temp(:, 1));
    
    all_trans_mat_rs_con(ref_sub, :, :) = trans_mat_rs_con;
    all_stationary_p_rs_con(ref_sub, :) = stationary_p_rs_con;
end
% figure;
% b = bar(stationary_p_rs_con);
% b.FaceColor = 'flat';
% b.CData(1:4, :) = cmap4(1:4, :);
% ylim([0, 0.6])

%% schz
configAffParams1.kNN = 220;
configAffParams2.kNN = 220;
[dm_schz, ~, embed_schz, lambda1_schz, lambda2_schz, sigma1_schz, sigma2_schz]...
    = calc2sDM(all_task_schz, n_dim, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

all_trans_mat_rs_schz = zeros(n_sub_schz, n_cluster, n_cluster);
all_stationary_p_rs_schz = zeros(n_sub_schz, n_cluster);

for ref_sub = 1 : n_sub_schz
    % generate extension embedding
    k_ext = 150;
    psi1 = zeros(n_tp_rest, n_dim*n_sub_schz);
    configAffParams3 = configAffParams1;
    configAffParams3.kNN = k_ext;
    all_rest_sync = zeros(n_region, n_tp_rest, n_sub_schz);
    
    for i = 1 : n_sub_schz
        disp(i)
        data_ind = zeros(n_region, n_tp_all);
        data_ind(:, train_range) = all_task_schz(:, :, i);
        data_ind(:, test_range) = all_rest_schz(:, :, i);
        if i ~= ref_sub
            [Y2, R] = brainSync(all_rest_schz(:, :, ref_sub)', data_ind(:, test_range)');
            data_ind(:, test_range) = Y2';
        end
        all_rest_sync(:, :, i) = data_ind(:, test_range);
        configAffParams3.sig = sigma1_schz(i);
        [K] = calcAffinityMat(data_ind, configAffParams3);
        K = K(test_range, train_range);
        K = K./sum(K, 2);
        psi1(:, (i-1)*n_dim+1:i*n_dim) = K*embed_schz(:, (i-1)*n_dim+1:i*n_dim)./lambda1_schz(:, i)';
    end
    
    % second round embedding
    
    embed_all = zeros(n_dim*n_sub_schz, n_tp_all);
    embed_all(:, train_range) = embed_schz';
    embed_all(:, test_range) = psi1';
    configAffParams3.sig = sigma2_schz;
    [K, ~] = calcAffinityMat(embed_all, configAffParams3);
    K = K(test_range, train_range);
    K = K./sum(K, 2);
    psi2 = K*dm_schz'./lambda2_schz';
    
    dm_schz_all = zeros(n_dim, n_tp_all);
    dm_schz_all(:, train_range) = dm_schz;
    dm_schz_all(:, test_range) = psi2';
    
    % get transition matrix and stationary distribution
    % get rs_kmeans_idx
    [~, I] = pdist2(dm_schz(1:3, :)', dm_schz_all(1:3, test_range)', 'euclidean','Smallest', 1);
    rs_kmeans_idx = kmeans_idx_schz(I);
    
    trans_mat_rs_schz = zeros(n_cluster, n_cluster);
    for i_tp = 1 : numel(test_range)-1
        %     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = rs_kmeans_idx(i_tp);
        c2 = rs_kmeans_idx(i_tp+1);
        trans_mat_rs_schz(c1, c2) = trans_mat_rs_schz(c1, c2) + 1;
        %     end
    end
    trans_mat_rs_schz = trans_mat_rs_schz./ sum(trans_mat_rs_schz, 2);
    [temp, ~] = eigs(trans_mat_rs_schz');
    stationary_p_rs_schz = temp(:, 1) / sum(temp(:, 1));
    all_trans_mat_rs_schz(ref_sub, :, :) = trans_mat_rs_schz;
    all_stationary_p_rs_schz(ref_sub, :) = stationary_p_rs_schz ;
end

%% plot
% figure;
% b = bar(stationary_p_rs_schz);
% b.FaceColor = 'flat';
% b.CData(1:4, :) = cmap4(1:4, :);
% ylim([0, 0.6])

% stationary distribution
figure;
offset = 0.1;
boxplot(all_stationary_p_rs_con, 'positions', (1:4)-offset, 'Colors', 'b', 'Widths', 0.2)
hold on
boxplot(all_stationary_p_rs_schz, 'positions', (1:4)+offset, 'Colors', 'r', 'Widths', 0.2)
hold off
ylim([0, 0.5])

% entropy
figure;
rs_entropy_control = zeros(77, 4);
for i_sub = 1 : 77
    for j_state = 1 : 4
        temp_pmf = all_trans_mat_rs_con(i_sub, j_state,:);
        temp_pmf(temp_pmf==0) = [];
        rs_entropy_control(i_sub, j_state) = -sum(temp_pmf.*log(temp_pmf));
    end
end
offset = 0.1;
boxplot(rs_entropy_control, 'positions', [1:4]-offset, 'Colors', 'b', 'Widths', 0.2)
hold on
rs_entropy_schz = zeros(44, 4);
for i_sub = 1 : 44
    for j_state = 1 : 4
        temp_pmf = all_trans_mat_rs_schz(i_sub, j_state,:);
        temp_pmf(temp_pmf==0) = [];
        rs_entropy_schz(i_sub, j_state) = -sum(temp_pmf.*log(temp_pmf));
    end
end
boxplot(rs_entropy_schz, 'positions', [1:4]+offset, 'Colors', 'r', 'Widths', 0.2)
% xlim([0.3, 5])
ylim([0.5, 1.4])

% Markov chain
% control
temp = squeeze(mean(all_trans_mat_rs_con(:,:,:), 1));
temp(temp<0.1) = 0; % removing small edges
if temp(1,4)==0 && temp(1,4)==0
    temp(1,4) = 0.0001;
end
mc = dtmc(temp,'StateNames',stateNames);
figure;
h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);

temp = temp';
h.LineWidth=log(temp(temp~=0)+1.3)*10;
h.NodeFontSize = 13;
h.EdgeFontSize = 10;

temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
colormap(temp_colormap)

% schz
temp = squeeze(mean(all_trans_mat_rs_schz(:,:,:), 1));
temp(temp<0.1) = 0; % removing small edges
% if temp(2,3)==0 && temp(3,2)==0
%     temp(2,3) = 0.0001;
% end
mc = dtmc(temp,'StateNames',stateNames);
figure;
h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);

temp = temp';
h.LineWidth=log(temp(temp~=0)+1.3)*10;
h.NodeFontSize = 13;
h.EdgeFontSize = 10;

temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
colormap(temp_colormap)
colorbar('off')