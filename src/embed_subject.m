load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')

% choose less subjects for quicker computation
data = data(:, 1:3020, :);
data = permute(data, [1,3,2]);

%% config the parameters
k = 100;
configAffParams1.dist_type = 'correlation';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 7;

%% embed the subjects
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/all_behav_relational.mat');


figure;

subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(3, :), [], all_behav, 'filled'); % color subjects by behavior scores

subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), [], all_behav, 'filled');

[rho, pval] = corr(dm', all_behav);
[~, s_ind] = sort(abs(rho), 'descend');
rho = rho(s_ind); %keep the correlation sign
pval = pval(s_ind);

subplot(2,2,3);
i = 1;
x = dm(s_ind(i), :)';
plot(x, all_behav, '.'); 
hold on;
fit = polyfit(x, all_behav, 1);
plot(x, polyval(fit, x), 'r-')
title([num2str(s_ind(i)), 'dim, corr=',num2str(rho(i)),', pval=',num2str(pval(i))])
hold off;

subplot(2,2,4);
i = 2;
x = dm(s_ind(i), :)';
plot(x, all_behav, '.'); 
hold on;
fit = polyfit(x, all_behav, 1);
plot(x, polyval(fit, x), 'r-')
title([num2str(s_ind(i)), 'dim, corr=',num2str(rho(i)),', pval=',num2str(pval(i))])
hold off;


