% %% generate extension embedding
% 
% 
% % % choose less subjects for quicker computation
% % addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync')
% % 
% load('/Users/siyuangao/Working_Space/fmri/data/UCLA/rest199.mat')
% % load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_labels.mat')
% % load('/Users/siyuangao/Working_Space/fmri/data/colormaps/cmap12.mat')
% % load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap6.mat')
% % 
% % % algorithm unrelated unconfig
% % rng(665)
% 
% % zscore RS by region
% for i_sub = 1 : 199
%     all_signal(:, :, i_sub) = zscore(all_signal(:, :, i_sub), 0, 1);
% end

%% all_sub
all_trans_mat_rs = cell(n_cohorts, 1);
all_stationary_p_rs = cell(n_cohorts, 1);

for i_idx = 1 : n_cohorts
    n_sub = length(all_idx{i_idx});
    all_trans_mat_rs_temp = zeros(n_sub, n_cluster, n_cluster);
    all_stationary_p_rs_temp = zeros(n_sub, n_cluster);
    
    all_task_temp = ucla_task(:, :, all_idx{i_idx});
    all_rest_temp = all_signal(:, :, all_idx{i_idx});
    n_tp_task = size(all_task_temp, 2);
    n_tp_rest = size(all_rest_temp, 2);
    n_tp_all = n_tp_task + n_tp_rest;
    n_region = size(all_task_temp, 1);
    train_range = 1 : n_tp_task;
    test_range = n_tp_task+1 : n_tp_all;

    for ref_sub = 1 : n_sub
        disp(ref_sub)
        
        k_ext = 80;

        psi1 = zeros(n_tp_rest, n_dim*n_sub);
        configAffParams3 = configAffParams1;
        configAffParams3.kNN = k_ext;
        all_rest_sync = zeros(n_region, n_tp_rest, n_sub);

        for i = 1 : n_sub
            % brainsync
            data_ind = zeros(n_region, n_tp_all);
            data_ind(:, train_range) = all_task_temp(:, :, i);
            data_ind(:, test_range) = all_rest_temp(:, :, i);
            if i ~= ref_sub
                [Y2, R] = brainSync(all_rest_temp(:, :, ref_sub)', ...
                    data_ind(:, test_range)');
                data_ind(:, test_range) = Y2';
            end
            all_rest_sync(:, :, i) = data_ind(:, test_range);
            configAffParams3.sig = sigma1_all{i_idx}(i);
            [K] = calcAffinityMat(data_ind, configAffParams3);
            K = K(test_range, train_range);
            K = K./sum(K, 2);
            psi1(:, (i-1)*n_dim+1:i*n_dim) = K*embed_all{i_idx}(:, ...
                (i-1)*n_dim+1:i*n_dim)./lambda1_all{i_idx}(:, i)';
        end

        % second round embedding
        embed_temp = zeros(n_dim*n_sub, n_tp_all);
        embed_temp(:, train_range) = embed_all{i_idx}';
        embed_temp(:, test_range) = psi1';
        configAffParams3.sig = sigma2_all{i_idx};
        [K, ~] = calcAffinityMat(embed_temp, configAffParams3);
        K = K(test_range, train_range);
        K = K./sum(K, 2);
        psi2 = K*dm_all{i_idx}'./lambda2_all{i_idx}';

        dm_temp_all = zeros(n_dim, n_tp_all);
        dm_temp_all(:, train_range) = dm_all{i_idx};
        dm_temp_all(:, test_range) = psi2';

        % get transition matrix and stationary distribution
        % get rs_kmeans_idx
        [~, I] = pdist2(dm_all{i_idx}(1:3, :)', ...
            dm_temp_all(1:3, test_range)', 'euclidean','Smallest', 1);
        rs_kmeans_idx = kmeans_idx_all{i_idx}(I);

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
    all_trans_mat_rs{i_idx} = all_trans_mat_rs_temp;
    all_stationary_p_rs{i_idx} = all_stationary_p_rs_temp;
end


%% plot
% figure;
% b = bar(stationary_p_rs_schz);
% b.FaceColor = 'flat';
% b.CData(1:4, :) = cmap4(1:4, :);
% ylim([0, 0.6])

% stationary distribution

% entropy
figure;
rs_entropy_all = cell(n_cohorts, 1);
for i_idx = 1 : n_cohorts
    n_sub = length(all_idx{i_idx});
    rs_entropy_temp = zeros(n_sub, n_cluster);
    for i_sub = 1 : n_sub
        for j_state = 1 : n_cluster
            temp_pmf = all_trans_mat_rs{i_idx}(i_sub, j_state,:);
            temp_pmf(temp_pmf==0) = [];
            rs_entropy_temp (i_sub, j_state) = -sum(temp_pmf.*log(temp_pmf));
        end
    end
%     offset = 0.1;
%     boxplot(rs_entropy, 'positions', [1:n_cluster]-offset, 'Colors', 'b', 'Widths', 0.2)
%     hold on
    rs_entropy_all{i_idx} = rs_entropy_temp;
end

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