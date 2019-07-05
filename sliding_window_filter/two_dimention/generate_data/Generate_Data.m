function [data] = Generate_Data()
%  输入：
%　输出：生成机器人位置，地标点，观测数据,里程计数据．
radius = 2;
theta_delta = 0.1;
num_pose = 5;
data.Ri = 0.000;
data.Qi = 0.000;

landmarks = [];
poses = [];
observations = [];
odoms = [];

%% generate landmarks;
theta = 0;
max_theta = theta_delta*num_pose;
while (theta < max_theta)
    x = radius * cos(theta);
    y = radius * sin(theta);
    landmarks = cat(1, landmarks, [x,y]);
    theta = theta + theta_delta;
end
%% generate poses;
theta = theta_delta;
while (theta < max_theta + theta_delta)
    x = 3.0/4.0*radius * cos(theta);
    y = 3.0/4.0*radius * sin(theta);
    poses = cat(1, poses, [x,y]);
    theta = theta + theta_delta;
end

%% generate observations;
size_pose = size(poses,1);
size_landmark = size(landmarks,1);
% 在每一个位置上，观察到三个ｌａｎｄｍａｒｋ
id_pose = 1;
id_landmark = 1;
observations_matrix = zeros(size_pose, size_landmark);
while (id_pose <= size_pose)
    id_landmark = id_pose;
    num_obsv = 3;
    id_obsv = 1;
    while (id_obsv <= num_obsv)
        id_obsv = id_obsv + 1;
        if (id_landmark > size_landmark)
            id_landmark = id_landmark - size_landmark +1;
        end
        pose = poses(id_pose,:);
        landmark = landmarks(id_landmark,:);
        obs = landmark - pose;
        noise = normrnd(0,sqrt(data.Qi),1,1);
        observations_matrix(id_pose, id_landmark) = obs(1);
        observations = cat(1, observations, [id_pose, id_landmark, obs + noise]);
        id_landmark = id_landmark + 1;
    end
    id_pose = id_pose + 1;
end

%% generate the odoms
for id_pose = 1:size_pose-1
    pose1 = poses(id_pose, :);
    pose2 = poses(id_pose + 1, :);
    odom = pose2 - pose1;
    noise = normrnd(0,sqrt(data.Qi),1,1);
    odoms = cat(1, odoms, odom + noise);
end


%% plot
figure;
scatter(landmarks(:,1),landmarks(:,2));
hold on; scatter(poses(:,1),poses(:,2));
legend('landmarks','poses');
axis('equal');

figure;
spy(observations_matrix);
xlabel('landmarks');ylabel('poses');

data.landmarks = landmarks;
data.odoms = odoms;
data.poses = poses;
data.observations = observations;
data.initpose = poses(1,:);
end

