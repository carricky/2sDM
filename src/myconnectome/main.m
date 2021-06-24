addpath('/Users/siyuangao/Working_Space/fmri/data/myconnectome/');
read_data;

addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/');
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync/');

%% parse dimensions
num_parc = size(data, 1);
num_time = size(data, 2);
num_sess = size(data, 3);

%% zscore data
for i_sess = 1 : num_sess
    data(:, :, i_sess) = zscore(data(:, :, i_sess), 0, 1);
end

%% BrainSync across sessions
ref_sess = 3;
for i_sess = 1 : num_sess
    if i_sess == ref_sess
        continue;
    end
    [Y2, R] = brainSync(data(:, :, ref_sess)', data(:, :, i_sess)');
    data(:, :, i_sess) = Y2';
end

%% config parameters
k = 100;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 10;

%% generate training embedding
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

s = 25*ones(num_time, 1);
c = 1:num_time;
figure; scatter3(dm(1,:), dm(2,:), dm(3,:), s, c, '.');