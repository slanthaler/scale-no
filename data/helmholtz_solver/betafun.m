function [f] =betafun(x,y)
if (1-x)*(1-y)*x*y == 0
    f = 1;
else 
    f = 0;
end
end