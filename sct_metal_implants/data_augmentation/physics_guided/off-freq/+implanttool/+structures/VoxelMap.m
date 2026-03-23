% A data structure to keep track of and manipulate a 3D voxel map, along
% with its scale and positions of each voxel.
classdef VoxelMap < handle
    properties
        V(:,:,:) = zeros(0,0,0);
        N(1,:) {int16} = zeros(1,3);
        voxelSize(1,:) {double} = ones(1,3);
        X(:,:,:);
        Y(:,:,:);
        Z(:,:,:);
        axisLabels = ["x [mm]", "y [mm]", "z [mm]"];
        valueLabel = '';
        plottingScale = 1;
    end
    methods
        function obj = VoxelMap(V,voxelSize)
            arguments
                V(:,:,:);
                voxelSize(1,:) {double} = ones(1,3);
            end
            obj.V = V;
            obj.N = size(V);
            if length(voxelSize) == 3
                obj.voxelSize = voxelSize;
            elseif length(voxelSize) == 1
                obj.voxelSize = repmat(voxelSize,[1,3]);
            end
        end

        function [] = calculateCoordinates(obj)
            extent = obj.extent() - obj.voxelSize;

            x = linspace(-extent(1)./2,extent(1)./2,obj.N(1));
            y = linspace(-extent(2)./2,extent(2)./2,obj.N(2));
            z = linspace(-extent(3)./2,extent(3)./2,obj.N(3));

            [X,Y,Z] = meshgrid(x,y,z);
            obj.X = permute(X,[2,1,3]);
            obj.Y = permute(Y,[2,1,3]);
            obj.Z = permute(Z,[2,1,3]);
        end

        function [X,Y,Z] = coordinates(obj)
            if isempty(obj.X) || isempty(obj.Y) || isempty(obj.Z)
                obj.calculateCoordinates();
            end

            X = obj.X;
            Y = obj.Y;
            Z = obj.Z;
        end

        function coord = dimToCoordinate(obj, dim)
            if isempty(obj.X) || isempty(obj.Y) || isempty(obj.Z)
                obj.calculateCoordinates();
            end

            if dim == 1
                coord = obj.X;
            elseif dim == 2
                coord = obj.Y;
            elseif dim == 3
                coord = obj.Z;
            end
        end

        function [] = copyCoordinates(obj,otherObj)
            arguments
                obj implanttool.structures.VoxelMap;
                otherObj implanttool.structures.VoxelMap;
            end
            [obj.X,obj.Y,obj.Z] = otherObj.coordinates();
        end

        function r = extent(obj)
            r = obj.N.*obj.voxelSize;
        end

        function r = valueRange(obj)
            r = [min(obj.V,[],'all'), max(obj.V,[],'all')];
        end

        function slice = getSlice(obj,pos,dim)
            import implanttool.tools.otherDims
            others = otherDims(dim);
            
            extent = obj.extent();
            idx = int16((pos./extent(dim) + 0.5) * obj.N(dim));
            
            values = squeeze(subsref(obj.V,substruct('()', [repelem({':'},dim-1) idx repelem({':'},ndims(obj.V)-dim)])));
            slice = implanttool.structures.PixelMap(values,obj.voxelSize(others));
            slice.axisLabels = obj.axisLabels(others);
            slice.valueLabel = obj.valueLabel;
            slice.plottingScale = obj.plottingScale;

            X = obj.dimToCoordinate(others(1));
            Y = obj.dimToCoordinate(others(2));
            
            slice.X = squeeze(subsref(X,substruct('()', [repelem({':'},dim-1) idx repelem({':'},ndims(obj.V)-dim)])));
            slice.Y = squeeze(subsref(Y,substruct('()', [repelem({':'},dim-1) idx repelem({':'},ndims(obj.V)-dim)])));
        end

        function [] = plotSlice(obj,pos,dim,ax)
            arguments
                obj;
                pos;
                dim;
                ax = gca;
            end
            obj.getSlice(pos,dim).plot(ax);
        end

        function [] = plot3D(obj)
            ax = gca;

            [X,Y,Z] = obj.coordinates();

            v_abs_max = max(abs(obj.V),[],'all');
            v = linspace(-v_abs_max, v_abs_max,21);
            c = parula(length(v));
            alpha = 2 / length(v);

            for i = length(v):-1:1
                p = patch (ax, isosurface(X,Y,Z,obj.V,v(i)));
                p.FaceColor = c(i,:);
                p.FaceAlpha = alpha;
                p.EdgeColor = 'none';
                p.DisplayName = compose("%.2f kHz",v(i) / (obj.plottingScale));
            end

            ax.DataAspectRatio = [1,1,1];
            ax.View = [-37.5,30];
            camlight(ax);
            ax.XLabel.String = obj.axisLabels(1);
            ax.YLabel.String = obj.axisLabels(2);
            ax.ZLabel.String = obj.axisLabels(3);
            legend(ax);
        end

        function [] = resample(obj, voxelSize)
            if length(voxelSize) == 1
                voxelSize = repmat(voxelSize,[1,3]);
            end

            [X,Y,Z] = obj.coordinates();
            X = permute(X,[2,1,3]);
            Y = permute(Y,[2,1,3]);
            Z = permute(Z,[2,1,3]);
            V = permute(obj.V,[2,1,3]);

            obj.N = ceil(obj.extent()./voxelSize);
            obj.voxelSize = voxelSize;

            obj.calculateCoordinates();
            [Xq,Yq,Zq] = obj.coordinates();
            Xq = permute(Xq,[2,1,3]);
            Yq = permute(Yq,[2,1,3]);
            Zq = permute(Zq,[2,1,3]);

            Vq = interp3(X,Y,Z,V,Xq,Yq,Zq,'nearest',0);
%             Vq = griddata(X,Y,Z,double(V),Xq,Yq,Zq);
            obj.V = permute(Vq,[2,1,3]);
        end

        function [] = exportASCII(obj, filename)
            [X,Y,Z] = obj.coordinates();
            
            X = reshape(X,[],1);
            Y = reshape(Y,[],1);
            Z = reshape(Z,[],1);
            V = reshape(obj.V,[],1);

            fid = fopen(filename,'w');
            fprintf(fid,'X[mm]\tY[mm]\tZ[mm]\tB_Z[G]\n');
            fclose(fid);


            dlmwrite(filename,[X,Y,Z,V],'-append','delimiter','\t');
        end

        function [] = zeroPadding(obj, e)
            arguments
                obj implanttool.structures.VoxelMap;
                e
            end

            if isequal(size(e), [1 1])
                e = repmat(e,[3,2]);
            elseif isequal(size(e), [1 3])
                e = [e' e'];
            elseif isequal(size(e), [3 2])
            end
            obj.V = cat(3, ...
                    cat(2, ...
                        cat(1, zeros([e(1,1),e(2,1),e(3,1)]),     zeros([obj.N(1),e(2,1),e(3,1)]),   zeros([e(1,2),e(2,1),e(3,1)])), ...
                        cat(1, zeros([e(1,1),obj.N(2),e(3,1)]),   zeros([obj.N(1),obj.N(2),e(3,1)]), zeros([e(1,2),obj.N(2),e(3,1)])), ...
                        cat(1, zeros([e(1,1),e(2,2),e(3,1)]),     zeros([obj.N(1),e(2,2),e(3,1)]),   zeros([e(1,2),e(2,2),e(3,1)])) ...
                    ), ...
                    cat(2, ...
                        cat(1, zeros([e(1,1),e(2,1),obj.N(3)]),   zeros([obj.N(1),e(2,1),obj.N(3)]),  zeros([e(1,2),e(2,1),obj.N(3)])), ...
                        cat(1, zeros([e(1,1),obj.N(2),obj.N(3)]), obj.V,                              zeros([e(1,2),obj.N(2),obj.N(3)])), ...
                        cat(1, zeros([e(1,1),e(2,2),obj.N(3)]),   zeros([obj.N(1),e(2,2),obj.N(3)]),  zeros([e(1,2),e(2,2),obj.N(3)])) ...
                    ), ...
                    cat(2, ...
                        cat(1, zeros([e(1,1),e(2,1),e(3,2)]),     zeros([obj.N(1),e(2,1),e(3,2)]),    zeros([e(1,2),e(2,1),e(3,2)])), ...
                        cat(1, zeros([e(1,1),obj.N(2),e(3,2)]),   zeros([obj.N(1),obj.N(2),e(3,2)]),  zeros([e(1,2),obj.N(2),e(3,2)])), ...
                        cat(1, zeros([e(1,1),e(2,2),e(3,2)]),     zeros([obj.N(1),e(2,2),e(3,2)]),    zeros([e(1,2),e(2,2),e(3,2)])) ...
                    ) ...
                );

            obj.N = size(obj.V);
            obj.X = ones(0,0,0);
            obj.Y = ones(0,0,0);
            obj.Z = ones(0,0,0);
        end

        function rotated = rotate(obj,R)
            [Xq,Yq,Zq] = obj.coordinates();

%             R = inv(R);

            X = R(1,1)*Xq + R(1,2)*Yq + R(1,3)*Zq;
            Y = R(2,1)*Xq + R(2,2)*Yq + R(2,3)*Zq;
            Z = R(3,1)*Xq + R(3,2)*Yq + R(3,3)*Zq;

            rotated = implanttool.structures.VoxelMap(obj.V,obj.voxelSize);
            rotated.V = griddata(X,Y,Z,double(obj.V),Xq,Yq,Zq);
        end

        function translated = translate(obj,p)
            if isempty(obj.X) || isempty(obj.Y) || isempty(obj.Z)
                obj.calculateCoordinates();
            end
            
            translated = implanttool.structures.VoxelMap(obj.V,obj.voxelSize);
            translated.copyCoordinates(obj);

            translated.X = translated.X + p(1);
            translated.Y = translated.Y + p(2);
            translated.Z = translated.Z + p(3);
        end

    end
end
