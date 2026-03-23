% Inflates a volume by the specified radius (i.e. takes the Minkowski sum
% between the volume and a sphere of specified radius)
function inflated_grid = inflateVolume(grid, inflation_radius)
    % Check if the input grid is a 3D logical array
    if ~islogical(grid) || ndims(grid) ~= 3
        error('Input grid must be a 3D logical array.');
    end

    % Check if the inflation_radius is a positive scalar value
    if ~isscalar(inflation_radius) || inflation_radius <= 0
        error('Inflation radius must be a positive scalar value.');
    end

    % Create a spherical structuring element for dilation
    % Note: You can also use other types of structuring elements for different shapes of inflation.
    se = strel('sphere', inflation_radius);

    % Perform morphological dilation on the grid
    inflated_grid = imdilate(grid, se);
end
