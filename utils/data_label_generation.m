function [ucla_task, all_labels] = data_label_generation(flag_zscore)
    n_task = 4;
    dataset = cell(n_task, 1);
    dataset{1} = 'pamenc';
    dataset{2} = 'pamret';
    dataset{3} = 'scap';
    dataset{4} = 'taskswitch';
    
    ucla_task = [];
    for i_task = 1 : n_task
        load(['/Users/siyuangao/Working_Space/fmri/data/UCLA/', dataset{i_task}, '199.mat'])
        ucla_task = cat(2, ucla_task, all_signal);
    end
    
    n_subs = size(ucla_task, 3);
    if flag_zscore
        % zscore by regions
        for i_sub = 1 : n_subs
            ucla_task(:, :, i_sub) = zscore(ucla_task(:, :, i_sub), 0, 1);
        end
    end
    
    load('/Users/siyuangao/Working_Space/fmri/data/UCLA/task_label_files/pamenc_labels.mat')
    pamenc_labels = [zeros(1, 3), pamenc_labels(1:end-3)];
    all_labels = pamenc_labels;
    
    load('/Users/siyuangao/Working_Space/fmri/data/UCLA/task_label_files/pamret_labels.mat')
    pamret_labels = [zeros(1, 3), pamret_labels(1:end-3)];
    pamret_labels(pamret_labels~=0) = pamret_labels(pamret_labels~=0) + max(all_labels);
    all_labels = [all_labels, pamret_labels];
    
    load('/Users/siyuangao/Working_Space/fmri/data/UCLA/task_label_files/scap_labels.mat')
    scap_labels = [zeros(1, 3), scap_labels(1:end-3)];
    scap_labels(scap_labels~=0) = scap_labels(scap_labels~=0) + max(all_labels);
    all_labels = [all_labels, scap_labels];
    
    load('/Users/siyuangao/Working_Space/fmri/data/UCLA/task_label_files/taskswitch_labels.mat')
    taskswitch_labels = [zeros(1, 3), taskswitch_labels(1:end-3)];
    taskswitch_labels(taskswitch_labels~=0) = taskswitch_labels(taskswitch_labels~=0) + max(all_labels);
    all_labels = [all_labels, taskswitch_labels];
    
%     figure; imagesc(all_labels)
end