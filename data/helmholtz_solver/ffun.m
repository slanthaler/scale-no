function f=ffun(x,y)
center_x = 0.5;
center_y = 0.5;

if (x-center_x)^2+(y-center_y)^2<1/400
    f = 10000*exp(-1/(1-400*((x-center_x)^2+(y-center_y)^2)));
else
    f= 0;
end
end