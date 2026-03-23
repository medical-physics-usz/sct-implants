% A data structure to keep track of and manipulate a 3D voxel map with
% logicical values, along with its scale and positions of each voxel.
classdef Volume < implanttool.structures.VoxelMap
    methods
        function obj = Volume(V,threshold,voxelSize)
            arguments
                V(:,:,:);
                threshold;
                voxelSize(1,:) {double} = Volume().voxelSize;
            end
            import implanttool.structures.VoxelMap
            obj@implanttool.structures.VoxelMap(V,voxelSize);

            obj.V = V>threshold;
        end

        function r = valueRange(obj)
            r = [0,1];
        end

        function p = plotVolume(obj, color, alpha, ax)
            arguments
                obj implanttool.structures.Volume;
                color = 'red';
                alpha = 1;
                ax = gca;
            end
            [X,Y,Z] = obj.coordinates();
            p = patch(ax,isosurface(X,Y,Z,obj.V));
            p.FaceColor = color;
            p.FaceAlpha = alpha;
            p.EdgeColor = 'none';
            ax.DataAspectRatio = [1,1,1];
            ax.View = [-37.5,30];
            camlight(ax);
            ax.XLabel.String = obj.axisLabels(1);
            ax.YLabel.String = obj.axisLabels(2);
            ax.ZLabel.String = obj.axisLabels(3);
        end

        function [xidx,yidx,zidx] = trim(obj, buffer)
            arguments
                obj implanttool.structures.Volume;
                buffer = 1;
            end
            [x,y,z] = ind2sub(size(obj.V),find(obj.V));
            xmin = max(min(x) - buffer, 1);
            xmax = min(max(x) + buffer, obj.N(1));
            ymin = max(min(y) - buffer, 1);
            ymax = min(max(y) + buffer, obj.N(2));
            zmin = max(min(z) - buffer, 1);
            zmax = min(max(z) + buffer, obj.N(3));

%             mask = false(obj.N);
%             mask(xmin:xmax,ymin:ymax,zmin:zmax) = 1;
%             mask = [xmin:xmax,ymin:ymax,zmin:zmax];

            xidx = xmin:xmax;
            yidx = ymin:ymax;
            zidx = zmin:zmax;
            
            obj.V = obj.V(xmin:xmax,ymin:ymax,zmin:zmax);

            if ~(isempty(obj.X) || isempty(obj.Y) || isempty(obj.Z))
                obj.X = obj.X(xmin:xmax,ymin:ymax,zmin:zmax);
                obj.Y = obj.Y(xmin:xmax,ymin:ymax,zmin:zmax);
                obj.Z = obj.Z(xmin:xmax,ymin:ymax,zmin:zmax);
            end
            
            obj.N = size(obj.V);
        end

        function [] = zeroPadding(obj, e)
            arguments
                obj implanttool.structures.VoxelMap;
                e
            end
            zeroPadding@implanttool.structures.VoxelMap(obj, e);
            obj.V = obj.V > 0;
        end

        function rotated = rotateVolume(obj, R)
            temp = obj.rotate(R);
            rotated = implanttool.structures.Volume(temp.V,0.5,temp.voxelSize);
            rotated.copyCoordinates(temp);
        end

        function translated = translateVolume(obj, p)
            temp = obj.translate(p);
            translated = implanttool.structures.Volume(temp.V,0.5,temp.voxelSize);
            translated.copyCoordinates(temp);
        end
    end
end
