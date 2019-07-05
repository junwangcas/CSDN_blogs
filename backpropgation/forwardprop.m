function [a21, a22, a31, L] = forwardprop(a11,w21, w22, w31, w32)
a21 = a11*w21;
a22 = a11*w22;
a31 = a21*w31 + a22*w32;
L = 1/2*(a31 - 2)^2;
end

