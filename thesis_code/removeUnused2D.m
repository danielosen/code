function [p tri, edge] = removeDuplicates2D(p, tri, edge),
% function [p tri, edge] = removeDuplicates2D(p, tri, edge),
% 
% description:
%    removes all nodal points not appearing in the element array 
%
% arguments:
%    - p      nodal points  (matrix of size Npts x 4 )
%    - tri    element array (matrix of size (Nel x 3) or (Nel x 5) )
%    - edge   boundary elements (matrix of size (Nel x 2) or (Nel x 4) )
%   
% returns:
%    - p      the stripped down nodal array (smaller size)
%    - tri    updates nodal indexes of the element array (updated values)
%    - edge   updates nodal indexes of the boundary elements (updated values)

% author: Kjetil A. Johannessen
% last edit: November 2013

tmp  = tri(:,1:3);
N    = size(p,1);
used = zeros(N,1);
used(tmp(:)) = 1;
uu = sort(find(used == 1)); 

offset = zeros(1,N);
totOffset = 0;
k = 1;
for i=1:N,
	if(k <= numel(uu) && uu(k) ~= i),	
		totOffset = totOffset + 1;
	else
		k = k + 1;
	end
	offset(i) = totOffset;
end
p = p(uu,:);
Nel = size(tri, 1);
for i=1:Nel,
	tri(i,1:3) = tri(i,1:3) - offset(tri(i,1:3));
end

Nel = size(edge, 1);
for i=1:Nel,
	edge(i,1:2) = edge(i,1:2) - offset(edge(i,1:2));
end

