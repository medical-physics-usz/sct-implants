% Returns linear index of item located at vec in array A.
function idx = vec2idx(vec,A)
    if ndims(A)==1
        idx = vec;
    elseif ndims(A)==2
        idx = sub2ind(size(A),vec(1),vec(2));
    elseif ndims(A)==3
        idx = sub2ind(size(A),vec(1),vec(2),vec(3));
    end
end
