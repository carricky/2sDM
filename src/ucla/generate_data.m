n_task = 6;
dataset = cell(n_task, 1);
dataset{1} = 'bart';
dataset{2} = 'pamenc';
dataset{3} = 'pamret';
dataset{4} = 'stopsignal';
dataset{5} = 'scap';
dataset{6} = 'taskswitch';

all_task = [];
label = [];
for i = 1 : n_task
    load(['/Users/siyuangao/Working_Space/fmri/data/UCLA/', dataset{i}, '199.mat'])
    all_task = cat(2, all_task, all_signal);
    label = [label; i*ones(size(all_signal, 2), 1)];
end


n_subs = size(all_task, 3);
for i = 1 : n_subs
    all_task(:, :, i) = zscore(all_task(:, :, i), 0, 1);
end
