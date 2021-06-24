rs_entropy = zeros(30, 4);
stationary_p_rs = zeros(30, 4);
% generate extension embedding
parfor ref_sub = 1 : 30
    disp(ref_sub)
    psi1 = zeros(num_t_test, n_d*num_s);
    configAffParams3 = configAffParams1;
    configAffParams3.kNN = 500;
    for i = 1 : num_s
        data_ind = zeros(num_r, num_t_all);
        data_ind(:, train_range) = train_data(:, :, i);
        data_ind(:, test_range) = test_data(:, :, i);
        if i ~= ref_sub
            [Y2, R] = brainSync(test_data(:, :, ref_sub)', data_ind(:, test_range)');
            data_ind(:, test_range) = Y2';
        end
        configAffParams3.sig = sigma1(i);
        [K] = calcAffinityMat(data_ind, configAffParams3);
        K = K(test_range, train_range);
        K = K./sum(K, 2);
        psi1(:, (i-1)*n_d+1:i*n_d) = K*embed(:, (i-1)*n_d+1:i*n_d)./lambda1(:, i)';
    end
    
    % second round embedding 
    embed_all = zeros(n_d*num_s, num_t_all);
    embed_all(:, train_range) = embed';
    embed_all(:, test_range) = psi1';
    configAffParams3.sig = sigma2;
    [K, ~] = calcAffinityMat(embed_all, configAffParams3);
    K = K(test_range, train_range);
    K = K./sum(K, 2);
    psi2 = K*dm'./lambda2';
    
    dm_all = zeros(n_d, num_t_all);
    dm_all(:, train_range) = dm;
    dm_all(:, test_range) = psi2';

    % RS transition
    % get rs_kmeans_idx
    [~, I] = pdist2(dm(1:4, :)', dm_all(1:4, test_range)', 'euclidean','Smallest', 1);
    rs_kmeans_idx = kmeans_idx(I);
    kmeans_idx_all = [kmeans_idx; rs_kmeans_idx];
    
    temporal_idx = find(true_label_all(1:num_t_all)==9);
    trans_mat_rs = zeros(n_cluster, n_cluster);
    for i_tp = 1 : numel(temporal_idx)-1
        c1 = kmeans_idx_all(temporal_idx(i_tp));
        c2 = kmeans_idx_all(temporal_idx(i_tp+1));
        trans_mat_rs(c1, c2) = trans_mat_rs(c1, c2) + 1;
    end
    trans_mat_rs = trans_mat_rs./ sum(trans_mat_rs, 2);
    [temp, ~] = eigs(trans_mat_rs');
    stationary_p_rs(ref_sub, :) = temp(:, 1) / sum(temp(:, 1));
    
%     temp_mean = (1:4) * stationary_p_rs(ref_sub, :);
%     temp_m = ((1:4) - temp_mean).^3 * stationary_p_rs(ref_sub, :);
%     temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p_rs(ref_sub, :)'));
%     skew_rs = temp_m / temp_std^3;
%     
%     % entropy transition calculation
%     temp_trans_mat = trans_mat_rs;
%     for i_state = 1 : 4
%         temp_pmf = temp_trans_mat(i_state,:);
%         temp_pmf(temp_pmf==0) = [];
%         rs_entropy(ref_sub, i_state) = -sum(temp_pmf.*log(temp_pmf));
%     end
    
end

%% bar or boxplot
% figure;bh = boxplot(stationary_p_rs, 'Colors', cmap4(1:4, :));
% set(bh, 'LineWidth', 2)
% ylim([0, 0.6])

figure;b=bar(mean(stationary_p_rs));
b.FaceColor = 'flat';
b.CData(1:4, :) = cmap4(1:4, :);
ylim([0, 0.6])
hold on

CI = zeros(4, 2);
for i = 1 : 4
    SEM = std(stationary_p_rs(:, i))/sqrt(length(stationary_p_rs(:, i))); % Standard Error
    ts = tinv([0.025  0.975],length(stationary_p_rs(:, i))-1);      % T-Score
    CI(i, :) = ts*SEM;                      % Confidence Intervals
end

er = errorbar(1:4,mean(stationary_p_rs),CI(:, 1), CI(:, 2), 'CapSize', 28);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold off