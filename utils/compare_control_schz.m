%% schz
mean_conn_schz = zeros(268, 268);
count = 0;
for i = idx_schz
    temp_conn = corr(all_signal(:,:,i)');
    if sum(isnan(temp_conn(:))) == 0
        mean_conn_schz = mean_conn_schz + temp_conn;
        count = count + 1;
    end
end
mean_conn_schz = mean_conn_schz / count;

%% CTL
mean_conn_ctl = zeros(268, 268);
count = 0;
for i = idx_control
    temp_conn = corr(all_signal(:,:,i)');
    if sum(isnan(temp_conn(:))) == 0
        mean_conn_ctl = mean_conn_ctl + temp_conn;
        count = count + 1;
    end
end
mean_conn_ctl = mean_conn_ctl / count;

figure; 
subplot(2,1,1);imagesc(mean_conn_schz)
subplot(2,1,2);imagesc(mean_conn_ctl)

d = mean_conn_schz - mean_conn_ctl;
figure; 
imagesc(d.*(abs(d)>0.1))

% load('/Users/siyuangao/Working_Space/fmri/data/map268.mat')
d = d(map268(:,2), map268(:,2));

figure; 
imagesc(d.*(abs(d)>0.1))
