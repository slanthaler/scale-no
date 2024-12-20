function idx = loc2glo(N, m, n, i)
% N: number of elements in one direction
if i <= 2 % bottom
    idx = (N+1)*(n-1) + m + i - 1;
else % top 
    idx = (N+1)*n + m + 4-i;
end

end

