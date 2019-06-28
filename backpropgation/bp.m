a = [1,0,0,0];
w = [0.1,0.1,0.1,0.1];
w_list = w;
L_list = [];
i = 0;
while i < 100
dL_w = [0, 0, 0, 0];
[a(2), a(3), a(4), L] = forwardprop(a(1), w(1), w(2), w(3),w(4));
[dL_w(1), dL_w(2),dL_w(3),dL_w(4)]= backprop(a, w);
w = update_w(dL_w, w);
L_list = cat(1, L_list, L);
i = i + 1;
end
plot(1:length(L_list),L_list);