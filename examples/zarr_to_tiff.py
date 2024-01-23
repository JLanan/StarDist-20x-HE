import zarr
from tifffile import imwrite, TiffWriter

z_label_path = r"\\10.99.68.54\Digital pathology image lib\SenNet JHU TDA Project\SN-LW-PA-P003-B1_SNP004\Spatial transcriptomics\SNP004-sec030\S18_36012_0030 Nuclei Segmentation\label.zarr"
tiff_out_path = r"\\10.99.68.54\Digital pathology image lib\SenNet JHU TDA Project\SN-LW-PA-P003-B1_SNP004\Spatial transcriptomics\SNP004-sec030\S18_36012_0030 Nuclei Segmentation\instance_map.ome.tif"

z_label = zarr.open(z_label_path, mode='r')
label = z_label[:, :]  # Converts to numpy array
# pyramid_levels = [1, 2, 4, 8, 16, 32]
# imwrite(tiff_out_path, label)


subresolutions = 2
pixelsize = 0.4416  # micrometer
with TiffWriter(tiff_out_path, bigtiff=True) as tif:
    metadata={
        # 'axes': 'TCYXS',
        'SignificantBits': 32,
        # 'TimeIncrement': 0.1,
        # 'TimeIncrementUnit': 's',
        'PhysicalSizeX': pixelsize,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': pixelsize,
        'PhysicalSizeYUnit': 'µm',
        # 'Channel': {'Name': ['Channel 1', 'Channel 2']},
        # 'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
    }
    options = dict(
        photometric='minisblack',
        tile=(128, 128),
        # compression='jpeg',
        # resolutionunit='CENTIMETER',
        # maxworkers=2
    )
    tif.write(
        label,
        subifds=subresolutions,
        resolution=(1e4 / pixelsize, 1e4 / pixelsize),
        metadata=metadata,
        **options
    )
    # write pyramid levels to the two subifds
    # in production use resampling to generate sub-resolution images
    for level in range(subresolutions):
        mag = 2 ** (level + 1)
        tif.write(
            label[::mag, ::mag],
            subfiletype=1,
            resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
            **options
        )
    # add a thumbnail image as a separate series
    # it is recognized by QuPath as an associated image
    # thumbnail = (label[0, 0, ::8, ::8] >> 2).astype('uint32')
    # tif.write(thumbnail, metadata={'Name': 'thumbnail'})
