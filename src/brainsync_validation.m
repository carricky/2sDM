load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')

addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync/');

% choose less subjects for quicker computation
data = data(:, :, 1:30);
test_range = 3021:4220;
train_range = 1:3020;
num_t = max(test_range);
num_s = size(data, 3);
data = data(:, 1:num_t, :);
num_r = size(data, 1);

%% main
sub_dis = zeros(num_s);

corr_before  = zeros(num_r, num_s);
corr_after = zeros(num_r, num_s);
corr_before_mean = zeros(num_s,1);
corr_after_mean = zeros(num_s,1);
for ref_sub = 1:30
    disp(ref_sub)
    data_new = data;
    idx = [1:ref_sub-1, ref_sub+1:num_s];
    %% correlation before brainsync
    for i_region = 1 : num_r
        temp = corr(squeeze(data(i_region, test_range, :)));
        corr_before(i_region, :) = temp(ref_sub, :);
    end
    corr_before_mean(ref_sub) = mean(mean(corr_before(:,idx)));
    %% BrainSync data
    for i = 1 : num_s
        disp(i)
        if i ~= ref_sub
            [Y2, R] = brainSync(data(:, test_range, ref_sub)', data(:, test_range, i)');
            data_new(:, test_range, i) = Y2';
            sub_dis(ref_sub, i) = norm(data(:, test_range, ref_sub) - Y2', 'fro');
        end
    end
    
    for i_region = 1 : num_r
        temp = corr(squeeze(data_new(i_region, test_range, :)));
        corr_after(i_region, :) = temp(ref_sub, :);
    end
    corr_after_mean(ref_sub) = mean(mean(corr_after(:,idx)));
end

