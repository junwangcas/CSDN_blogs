function [ data ] = generate_g(data)
% generate error functions;
%% prepare gz;
gzs = [];
for i = 1:size(data.observations,1)
    obs = data.observations(i,:);
    z = obs(3:4);
    h_x = data.landmarks(obs(2),:) - data.poses(obs(1),:);
    gz = z - h_x;
    gzs = cat(1, gzs, gz');
end
data.g.gzs = gzs;

%% prepare gf;
gfs = [];
for i = 1:size(data.odoms,1)
    odom = data.odoms(i,:);
    f_x = data.poses(i+1,:) - data.poses(i,:);
    g_f = odom - f_x;
    gfs = cat(1, gfs, g_f');
end
data.g.gfs = gfs;
%histogram(gfs);

%% prepare init
ginit = [];
ginit = (data.initpose - data.poses(1,:))';
data.g.ginit = ginit;
end

