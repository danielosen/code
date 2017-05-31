%Build Node Connection Matrix for NonLocal Method
%Optimized with triplets storage for sparsity
%Author: Daniel Osen
function [C] = build_nodeconnection_nonlocal(pos_t,N),
%C = zeros(12,3*N);
I = 1:12;
J = zeros(1,12);
X = ones(1,12);
    for i=1:12
        %The indexing gives pos(1)+0, pos(1)+1, pos(1)+2,
        %pos(2)+0,...,pos(4)+2 etc, referring to the running index.
        %C(i,pos_t(floor((i-1)/3+1))+mod(i-1,3)) = 1;
        J(i) = pos_t(floor((i-1)/3+1))+mod(i-1,3);
    end
C = sparse(I,J,X,12,3*N,12);
end