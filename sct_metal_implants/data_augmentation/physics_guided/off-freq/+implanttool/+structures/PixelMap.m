% A data structure to keep track of and manipulate a 2D pixel map, along
% with its scale and positions of each pixel.
classdef PixelMap < handle
    properties
        V(:,:) = zeros(0,0);
        N(1,:) {int16} = zeros(1,2);
        pixelSize(1,:) {double} = ones(1,2);
        X(:,:);
        Y(:,:);
        axisLabels = ["x [mm]", "y [mm]"];
        valueLabel = '';
        plottingScale = 1;
    end
    methods
        function obj = PixelMap(V,pixelSize)
            arguments
                V(:,:);
                pixelSize(1,:) {double} = ones(1,2);
            end
            obj.V = V;
            obj.N = size(V);
            obj.pixelSize = pixelSize;
        end
        
        function [] = calculateCoordinates(obj)
            extent = obj.extent() - obj.pixelSize;
            
            x = linspace(-extent(1)./2,extent(1)./2,obj.N(1));
            y = linspace(-extent(2)./2,extent(2)./2,obj.N(2));
            
            [X,Y] = meshgrid(x,y);
            obj.X = X';
            obj.Y = Y';
        end
        
        function [X,Y] = coordinates(obj)
            if isempty(obj.X) || isempty(obj.Y)
                obj.calculateCoordinates();
            end
            
            X = obj.X;
            Y = obj.Y;
        end
        
        function coord = dimToCoordinates(obj, dim)
            if isempty(obj.X) || isempty(obj.Y)
                obj.calculateCoordinates();
            end
            
            if dim == 1
                coord = obj.X;
            elseif dim == 2
                coord = obj.Y;
            end
        end
        
        function [] = copyCoordinates(obj,otherObj)
            arguments
                obj implanttool.structures.PixelMap;
                otherObj implanttool.structures.PixelMap;
            end
            [obj.X,obj.Y] = otherObj.coordinates();
        end
        
        function r = extent(obj)
            r = obj.N.*obj.pixelSize;
        end
        
        function r = valueRange(obj)
            r = [min(obj.V,[],'all'), max(obj.V,[],'all')];
        end
        
        function im = plot(obj, ax, cmap)
            arguments
                obj;
                ax = gca;
                cmap = "gray";
            end
            [X,Y] = obj.coordinates();
            
            x = squeeze(X(:,1));
            y = squeeze(Y(1,:));
            
            im = image(x,y,obj.V'./obj.plottingScale,'CDataMapping','scaled','Parent',ax);
            
            ax.Colormap = colormap(cmap);
            ax.DataAspectRatio = [1,1,1];
            ax.PlotBoxAspectRatio = [obj.pixelSize,1];
            ax.YDir = 'normal';
            ax.XLabel.String = obj.axisLabels(1);
            ax.YLabel.String = obj.axisLabels(2);
        end

        function im = plotLogical(obj, ax, alpha)
            arguments
                obj;
                ax = gca;
                alpha = 1;
            end
            [X,Y] = obj.coordinates();

            c = ones([obj.N(2) obj.N(1) 3]);
            c(:,:,2:end) = 0;

            im = image(ax,squeeze(X(:,1)),squeeze(Y(1,:)),c,"AlphaData",obj.V'*alpha);
            
            ax.DataAspectRatio = [1,1,1];
            ax.PlotBoxAspectRatio = [obj.pixelSize,1];
            ax.YDir = 'normal';
            ax.XLabel.String = obj.axisLabels(1);
            ax.YLabel.String = obj.axisLabels(2);
        end
    end
end