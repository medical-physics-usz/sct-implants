% Connected-component labeling using "one component at a time" algorithm
function labelArray = identifyConnectedComponents(v)
    arguments
        v implanttool.structures.Volume;
    end
    import implanttool.tools.vec2idx;

    label = 1;
    queue = zeros(0,3);
    labelArray = zeros(v.N);
    for i=1:v.N(1)
        for j=1:v.N(2)
            for k=1:v.N(3)
                if ~labelArray(i,j,k) && v.V(i,j,k)
                    labelArray(i,j,k) = label;
                    queue = [i,j,k];
                    while size(queue,1) > 0
                        pixel = queue(1,:);
                        queue = queue(2:end,:);
                        neighbors = getNeighbors(v,pixel);
                        for n = 1:size(neighbors,1)
                            neighbor = neighbors(n,:);
                            if ~labelArray(vec2idx(neighbor,labelArray)) && v.V(vec2idx(neighbor,v.V))
                                labelArray(vec2idx(neighbor,labelArray)) = label;
                                queue = [queue; neighbor];
                            end
                        end
                    end
                    label = label+1;
                end
            end
        end
    end
end

function n = getNeighbors(v,coords)
    arguments
        v implanttool.structures.Volume;
        coords(1,3);
    end
    n = zeros(0,3);

    for di = [-1,0,1]
        for dj = [-1,0,1]
            for dk = [-1,0,1]
                if sum(abs([di,dj,dk])) == 1 && isWithinFov(v,coords + [di,dj,dk])
                    n = [n; coords + [di,dj,dk]];
                end
            end
        end
    end
end

function ret = isWithinFov(v,coords)
    arguments
        v implanttool.structures.Volume;
        coords;
    end
    if nnz(coords<1) || nnz(coords>v.N)
        ret = 0;
    else
        ret = 1;
    end
end