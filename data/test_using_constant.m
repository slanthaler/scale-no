%sanity check that the implementation gives correct solution
s = 1024;
[X,Y] = meshgrid(0:(1/(s-1)):1);
a=ones(s,s);
f = zeros(s,s);
%g = ones(s,s);
g =X.*Y;
p=solve_gwf(a,f,g);
mesh(p)
max(max(abs(p-g)))