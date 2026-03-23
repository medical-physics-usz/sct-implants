function slice = sliceSelection(df_slice, sliceCenter, sliceThickness, BW, slice_dim)
    arguments
        df_slice implanttool.structures.PixelMap;
        sliceCenter;
        sliceThickness;
        BW;
        slice_dim=2;
    end
    import implanttool.structures.PixelMap
    Z = df_slice.dimToCoordinates(slice_dim);

    values = sliceCenter - sliceThickness/2 < Z + df_slice.V ./BW * sliceThickness & Z + df_slice.V ./BW * sliceThickness < sliceCenter + sliceThickness/2;
    slice = PixelMap(values, df_slice.pixelSize);
end
