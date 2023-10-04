import zarr
import os
import tifffile as tiff
import numpy as np
from skimage import measure
OPENSLIDE_PATH = r"C:\Users\labuser\PyCharm Projects\StarDist\Whole Slide Imager\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def get_sections(wsi, z_label, horizontal, vertical, microns_per_pixel, width, height):
    is_20x = True
    h_slide, v_slide = int(horizontal / microns_per_pixel), int(vertical / microns_per_pixel)
    upper_slide, lower_slide = v_slide - height // 2, v_slide + height // 2
    left_slide, right_slide = h_slide - width // 2, h_slide + width // 2

    if int(wsi.properties['openslide.objective-power']) == 20:
        h_mask, v_mask = h_slide, v_slide
        upper_mask, lower_mask = upper_slide, lower_slide
        left_mask, right_mask = left_slide, right_slide
    else:
        is_20x = False
        h_mask, v_mask = round(h_slide / 2), round(v_slide / 2)
        upper_mask, lower_mask = v_mask - height // 2, v_mask + height // 2
        left_mask, right_mask = h_mask - width // 2, h_mask + width // 2

    if is_20x == True:
        slide_section = np.asarray(wsi.read_region((left_slide, upper_slide), 0, (width, height)).convert('RGB'))
    else:
        slide_section = np.asarray(wsi.read_region((left_slide - width // 2, upper_slide - height // 2), 1, (width, height)).convert('RGB'))

    mask_section = np.asarray(z_label[upper_mask:lower_mask, left_mask:right_mask])
    return slide_section, mask_section

def create_binary_threshold(intensity, mask_gray):
    # Create a new array with the modified values
    mask = np.zeros_like(mask_gray)  # Initialize with zeros
    indices = np.where(mask_gray == intensity)
    mask[indices] = 255
    return mask

def draw_contours(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contour_set = []
    uniques = np.unique(mask[mask != 0])
    for i in uniques:
        binary_thresh_msk = create_binary_threshold(i, mask)
        contour_set.append(measure.find_contours(binary_thresh_msk))

    for contours in contour_set:
        for contour in contours:
            image[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)] = [255, 255, 0]
    return image

def save_results(slide_section, overlay, save_folder, img_name_out_tissue, img_name_out_overlay):
    slide_section_name_out = os.path.join(save_folder, img_name_out_tissue)
    overlay_name_out = os.path.join(save_folder, img_name_out_overlay)
    tiff.imwrite(slide_section_name_out, slide_section)
    tiff.imwrite(overlay_name_out, overlay)
    return None


if __name__ == "__main__":
    wsi_path = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\he.ndpi"
    zarr_path = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\Nuclei Segmentations\Model_00 WSI_he\label.zarr"
    save_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\Nuclei Segmentations\Model_00 WSI_he"

    img_name_out_tissue = "ROI_01_tissue.TIFF"
    img_name_out_overlay = "ROI_01_overlay.TIFF"

    horizontal, vertical = 9040, 5450  # micron centerpoint using QuPath pointer
    microns_per_pixel = 0.2203  # from QuPath
    width, height = 1200, 800  # Desired pixel FoV for output image

    ####################################################################################################################
    ########################## You shouldn't have to touch anything below this line ####################################
    ####################################################################################################################

    wsi, z_label = OpenSlide(wsi_path), zarr.open(zarr_path, mode='r')
    slide_section, mask_section = get_sections(wsi, z_label, horizontal, vertical, microns_per_pixel, width, height)
    overlay = draw_contours(np.copy(slide_section), np.copy(mask_section))
    save_results(slide_section, overlay, save_folder, img_name_out_tissue, img_name_out_overlay)
