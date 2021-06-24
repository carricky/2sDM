function [stationary_dist, trans_mat, entropy] = calcDynamics(dm_train)
% generate extension embedding
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
    n_tp_task = size(all_task_temp, 2);
    n_tp_rest = size(all_rest_temp, 2);
    n_tp_all = n_tp_task + n_tp_rest;
    n_region = size(all_task_temp, 1);
    train_range = 1 : n_tp_task;
    test_range = n_tp_task+1 : n_tp_all;

    sigma1_iidx = sigma1_all;
    embed_iidx = embed_all;
    lambda1_iidx = lambda1_all;
    tic
    for ref_sub = 1 : n_sub
        disp(ref_sub)

        all_rest_ref = all_rest_temp(:, :, ref_sub);
        
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
for i_c = 1 : n_cluster
    [h, p_all_dist(i_c)] = ttest2(all_stationary_p_rs{1}(:, i_c), ...
        all_stationary_p_rs{2}(:, i_c));
end

%% entropy
% figure;
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
%     offset = 0.1;
%     boxplot(rs_entropy, 'positions', [1:n_cluster]-offset, 'Colors', 'b', 'Widths', 0.2)
%     hold on
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
for i_c = 1 : n_cluster
    [h, p_all_entropy(i_c)] = ttest2(rs_entropy_all{1}(:, i_c), ...
        rs_entropy_all{2}(:, i_c));
end
disp(p_all_entropy)
