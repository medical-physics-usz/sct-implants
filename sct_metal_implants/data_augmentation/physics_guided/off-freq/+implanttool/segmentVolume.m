% Segments a volume into its connected components. 
function volumeList = segmentVolume(v)
    import implanttool.tools.identifyConnectedComponents
    import implanttool.structures.Volume
    
    volumeList = [];
    labelArray = identifyConnectedComponents(v);

    numComponents = max(labelArray,[],'all');
    for i = 1:numComponents
        volumeList = [volumeList, Volume(labelArray==i,0,v.voxelSize)];
    end
end
