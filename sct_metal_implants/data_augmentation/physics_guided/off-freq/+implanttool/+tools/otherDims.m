% Given one of the three dimensions, the others are returned.
function others = otherDims(dim)
    if dim == 1
        others = [2,3];
    elseif dim == 2
        others = [1,3];
    elseif dim == 3
        others = [1,2];
    end
end