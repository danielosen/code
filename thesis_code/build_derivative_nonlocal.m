%Build tensor of derivatives for nonlocal method
function [tdiv] = build_derivative_nonlocal(cpt),
tdiv = zeros(1,12,3,3);
id = eye(3);
    for i=1:3
        for j=1:3
            for m=1:4
                tdiv(1,(3*(m-1)+1):(3*(m-1)+3),i,j) = 0.5*(cpt(j,m)*id(:,i)'+cpt(i,m)*id(:,j)')*id;
            end
        end
    end
end
