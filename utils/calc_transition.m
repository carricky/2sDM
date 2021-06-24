num_s = size(IDX_list_all, 1);
num_t = size(IDX_list_all, 2);
num_c = max(IDX_list_all(:));
trans_mat = zeros(num_c);
for i_sub = 1 : num_s
    for j_time = 2 : num_t
        trans_mat(IDX_list_all(i_sub, j_time-1), IDX_list_all(i_sub, j_time)) = trans_mat(IDX_list_all(i_sub, j_time-1), IDX_list_all(i_sub, j_time))+1;
    end
end
trans_mat = trans_mat ./ sum(trans_mat, 2);
figure; imagesc(trans_mat)
        