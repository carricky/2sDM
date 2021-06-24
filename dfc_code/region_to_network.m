function network_data = region_to_network(region_data, maps)
    n_sub = size(region_data, 3);
    n_time = size(region_data, 2);
    n_network = max(maps);
    network_data = zeros(n_network, n_time, n_sub);
    for i = 1 : n_network
        network_data(i, :, :) = mean(region_data(maps==i, :, :), 1);
    end 
end