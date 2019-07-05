data = Generate_Data();
data = reset_statevector(data);
%visualdata(data);

i = 0;
while (i < 10)
    data = generate_g(data);
    data = generate_jacobian(data);
    data.G_full = data.G.H + data.G.D + data.G.L;
    data.g_full = [data.g.gzs; data.g.gfs; data.g.ginit];
    delta_x = - data.G_full'*data.G_full\(data.G_full'*data.g_full);
    % update
    data = update_x(data,delta_x);
    %
    i = i+1
    sum(data.g_full)
    visualdata(data);
end


close all;



