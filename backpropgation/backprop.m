function [dL_w21, dL_w22, dL_w31, dL_w32] = backprop(a, w)
dL_w21 = (a(4) - 2) * a(1) * w(3);
dL_w22 = (a(4) - 2) * a(1) * w(4);
dL_w31 = (a(4) - 2) * a(2);
dL_w32 = (a(4) - 2) * a(3);
end

