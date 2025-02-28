% function a=afun(t,s)
% if abs(t-0.4)<0.015
%     a=10^4;
% else
%     a=1;
% end

% function a=afun(t,s)
%     epsilon =1/16;
%     pts1=0.5*epsilon:epsilon:1;
%     pts2=0.5*epsilon:epsilon:1;
%     [pts_x,pts_y]=meshgrid(pts1,pts2);
%     distance=max(abs(pts_x-t),abs(pts_y-s));
%     if (t-0.25)*(t-0.75)>0 || (s-0.25)*(s-0.75)>0
%         a=1;
%     elseif min(min(distance))<0.25*epsilon 
%         a=epsilon^2;
%     else
%         a=1;
%     end
% end

function a=afun(t,s)
    a=1;
end
