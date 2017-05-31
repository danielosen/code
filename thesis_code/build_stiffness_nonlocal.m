%Builds Constant Stiffness Matrix for NonLocal Method
function [M_tk] = build_stiffness_nonlocal(lambda,mu,tdiv,kdiv),
M_tk = zeros(12,12);
    for i=1:3
        M_tk = M_tk + (lambda+2*mu)*tdiv(1,:,i,i)'*kdiv(1,:,i,i);
        for j=1:3
            if i~=j
                M_tk = M_tk + lambda*tdiv(1,:,i,i)'*kdiv(1,:,j,j);
                M_tk = M_tk + 2*mu*tdiv(1,:,i,j)'*kdiv(1,:,i,j);
            end
        end
    end
end