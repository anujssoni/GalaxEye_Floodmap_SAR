import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from scipy.ndimage import uniform_filter, convolve, median_filter
from skimage.morphology import remove_small_objects, binary_opening, disk
import matplotlib.pyplot as plt
from rasterio.windows import Window
from sklearn.metrics import confusion_matrix, classification_report


pre_path = r"C:/GalaxEye/Preporcessed/Patna_S1A_GRD_Pre.tif"
post_path = r"C:/GalaxEye/Preporcessed/Patna_S1A_GRD_Post.tif"
aoi_path = r"C:/GalaxEye/layers/shp/AOI.shp"  
VH_BAND = 2                     
ASSUME_INPUT_DB = None          
WATER_THRESH_DB = -20.0        
DIFF_THRESH_DB = -2.5           
MIN_SIZE_RAW = 5                
MIN_SIZE_CLEAN = 20             
OUTPUT_DIR = r"C:/GalaxEye/outputs_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_SIZE = 1024
PADDING = 16  


def read_window_as_array(path, band_index, window, out_shape=None):
    with rasterio.open(path) as src:
        arr = src.read(band_index, window=window).astype('float32')
        meta = src.meta.copy()
        return arr, meta

def read_full_band(path, band_index, aoi_path=None):
    """Read whole band optionally clipped to AOI (returns array and meta)."""
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        if aoi_path:
            gdf = gpd.read_file(aoi_path)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            out_image, out_transform = mask(src, gdf.geometry, crop=True)
            band = out_image[band_index - 1].astype('float32')
            meta.update({
                "height": band.shape[0],
                "width": band.shape[1],
                "transform": out_transform,
                "crs": src.crs
            })
            return band, meta
        else:
            band = src.read(band_index).astype('float32')
            return band, meta

def save_uint8_mask(mask_bool, meta, path, nodata=0):
    meta2 = meta.copy()
    meta2.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=nodata, height=mask_bool.shape[0], width=mask_bool.shape[1])
    with rasterio.open(path, 'w', **meta2) as dst:
        dst.write(mask_bool.astype(np.uint8), 1)

def to_natural_from_db(db_arr):
    arr = np.array(db_arr, dtype='float32')
    arr[arr <= -100] = -100
    return np.power(10.0, arr / 10.0)

def to_db_from_linear(linear_arr):
    arr = np.array(linear_arr, dtype='float32')
    arr[arr <= 0] = 1e-6
    return 10.0 * np.log10(arr)

def detect_input_is_db(arr, assume=None):
    if assume is True: return True
    if assume is False: return False
    mn = np.nanmin(arr); mx = np.nanmax(arr)
    if mx <= 30 and mn >= -100 and mn < 0:
        return True
    if mn >= 0 and mx > 0:
        return False
    return True if mx <= 30 else False

def refined_lee_directional_tile(img_tile):
    
    mean3 = uniform_filter(img_tile, size=3, mode='reflect')
    mean_sq3 = uniform_filter(img_tile * img_tile, size=3, mode='reflect')
    var3 = mean_sq3 - mean3 * mean3

    
    offsets = [(-2, -2), (-2, 0), (-2, 2),
               ( 0, -2), ( 0, 0), ( 0, 2),
               ( 2, -2), ( 2, 0), ( 2, 2)]

    
    sample_mean = []
    sample_var = []
    for dy, dx in offsets:
        sm = np.roll(np.roll(mean3, shift=dy, axis=0), shift=dx, axis=1)
        sv = np.roll(np.roll(var3,  shift=dy, axis=0), shift=dx, axis=1)
        sample_mean.append(sm)
        sample_var.append(sv)
    sample_mean = np.stack(sample_mean, axis=0)  
    sample_var  = np.stack(sample_var, axis=0)

   
    g1 = np.abs(sample_mean[1] - sample_mean[7])
    g2 = np.abs(sample_mean[6] - sample_mean[2])
    g3 = np.abs(sample_mean[3] - sample_mean[5])
    g4 = np.abs(sample_mean[0] - sample_mean[8])
    gradients_stack = np.stack([g1, g2, g3, g4], axis=0)

    max_grad = np.max(gradients_stack, axis=0)
    gradmask = np.stack([(g1 == max_grad),
                         (g2 == max_grad),
                         (g3 == max_grad),
                         (g4 == max_grad)], axis=0)

    center = sample_mean[4]

    cond1 = (sample_mean[1] - center) > (center - sample_mean[7])
    cond2 = (sample_mean[6] - center) > (center - sample_mean[2])
    cond3 = (sample_mean[3] - center) > (center - sample_mean[5])
    cond4 = (sample_mean[0] - center) > (center - sample_mean[8])

    dir_masks = []
    dir_masks.append(cond1)
    dir_masks.append(cond2)
    dir_masks.append(cond3)
    dir_masks.append(cond4)
    dir_masks.append(np.logical_not(cond1))
    dir_masks.append(np.logical_not(cond2))
    dir_masks.append(np.logical_not(cond3))
    dir_masks.append(np.logical_not(cond4))
    dir_masks = np.stack(dir_masks, axis=0)  

   
    gradmask_dup = np.zeros_like(dir_masks, dtype=bool)
    gradmask_dup[0] = gradmask[0]; gradmask_dup[4] = gradmask[0]
    gradmask_dup[1] = gradmask[1]; gradmask_dup[5] = gradmask[1]
    gradmask_dup[2] = gradmask[2]; gradmask_dup[6] = gradmask[2]
    gradmask_dup[3] = gradmask[3]; gradmask_dup[7] = gradmask[3]

    valid_dirs = dir_masks & gradmask_dup

   
    directions_img = np.zeros(center.shape, dtype=np.uint8)
    for i in range(8):
        maski = valid_dirs[i]
        directions_img[maski] = i + 1

    
    eps = 1e-10
    sample_stats = sample_var / (sample_mean * sample_mean + eps)
    
    sorted_stats = np.sort(sample_stats, axis=0)
    sigmaV = np.mean(sorted_stats[:5, ...], axis=0)

    
    rect = np.zeros((7,7), dtype=float); rect[3:7, :] = 1.0; rect /= rect.sum()
    diag = np.array([
        [1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0],
        [1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0],
        [1,1,1,1,1,0,0],
        [1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1]
    ], dtype=float)
    diag /= diag.sum()

    kernels = []
    for i in range(4):
        kernels.append(np.rot90(rect, k=i))
        kernels.append(np.rot90(diag, k=i))
    

    img_sq = img_tile * img_tile
    dir_means = []
    dir_vars  = []
    for k in kernels:
        m = convolve(img_tile, k, mode='reflect')
        m2 = convolve(img_sq, k, mode='reflect')
        v = m2 - m*m
        dir_means.append(m)
        dir_vars.append(v)
    dir_means = np.stack(dir_means, axis=0)
    dir_vars  = np.stack(dir_vars, axis=0)

    
    dir_mean_selected = np.zeros_like(center, dtype=float)
    dir_var_selected  = np.zeros_like(center, dtype=float)
    for i in range(8):
        maski = (directions_img == (i+1))
        if np.any(maski):
            dir_mean_selected[maski] = dir_means[i][maski]
            dir_var_selected[maski]  = dir_vars[i][maski]

    nodir = (directions_img == 0)
    if np.any(nodir):
        dir_mean_selected[nodir] = mean3[nodir]
        dir_var_selected[nodir]  = var3[nodir]

    denom = (sigmaV + 1.0)
    denom[denom == 0] = 1e-8
    varX = (dir_var_selected - (dir_mean_selected * dir_mean_selected * sigmaV)) / denom
    dir_var_selected[dir_var_selected == 0] = 1e-8
    b = varX / dir_var_selected
    result = dir_mean_selected + b * (img_tile - dir_mean_selected)
    return result.astype('float32')

def process_full_image_in_tiles(nat_img, tile_size=TILE_SIZE, pad=PADDING):
    
    rows, cols = nat_img.shape
    out = np.zeros_like(nat_img, dtype='float32')

    for r0 in range(0, rows, tile_size):
        for c0 in range(0, cols, tile_size):
            r1 = min(r0 + tile_size, rows)
            c1 = min(c0 + tile_size, cols)
            
            r0_pad = max(0, r0 - pad)
            c0_pad = max(0, c0 - pad)
            r1_pad = min(rows, r1 + pad)
            c1_pad = min(cols, c1 + pad)
            tile = nat_img[r0_pad:r1_pad, c0_pad:c1_pad]
            
            filtered_tile = refined_lee_directional_tile(tile)
            
            r_start_in_tile = r0 - r0_pad
            c_start_in_tile = c0 - c0_pad
            r_end_in_tile = r_start_in_tile + (r1 - r0)
            c_end_in_tile = c_start_in_tile + (c1 - c0)
            out[r0:r1, c0:c1] = filtered_tile[r_start_in_tile:r_end_in_tile, c_start_in_tile:c_end_in_tile]
    return out


def main():
    print("Reading pre and post VH bands (full read & clip optional)...")
    pre_raw, meta = read_full_band(pre_path, VH_BAND, aoi_path)
    post_raw, _ = read_full_band(post_path, VH_BAND, aoi_path)

    print("Basic pre stats:", np.nanmin(pre_raw), np.nanmax(pre_raw), np.nanmean(pre_raw))
    print("Basic post stats:", np.nanmin(post_raw), np.nanmax(post_raw), np.nanmean(post_raw))

    
    input_is_db = detect_input_is_db(pre_raw, ASSUME_INPUT_DB)
    print("Input detected as:", "dB" if input_is_db else "linear")

    if input_is_db:
        pre_nat = to_natural_from_db(pre_raw)
        post_nat = to_natural_from_db(post_raw)
        pre_db = pre_raw.copy()
        post_db = post_raw.copy()
    else:
        pre_nat = pre_raw.copy()
        post_nat = post_raw.copy()
        pre_db = to_db_from_linear(pre_nat)
        post_db = to_db_from_linear(post_nat)

    
    print("Applying tile-based directional Refined-Lee filter...")
    pre_filt_nat = process_full_image_in_tiles(pre_nat, tile_size=TILE_SIZE, pad=PADDING)
    post_filt_nat = process_full_image_in_tiles(post_nat, tile_size=TILE_SIZE, pad=PADDING)

    print("Converting filtered results to dB...")
    pre_filt_db = to_db_from_linear(pre_filt_nat)
    post_filt_db = to_db_from_linear(post_filt_nat)

    diff_db = post_filt_db - pre_filt_db
    print("Filtered stats pre (dB):", np.nanmin(pre_filt_db), np.nanmax(pre_filt_db), np.nanmean(pre_filt_db))
    print("Filtered stats post (dB):", np.nanmin(post_filt_db), np.nanmax(post_filt_db), np.nanmean(post_filt_db))
    print("Diff stats (dB):", np.nanmin(diff_db), np.nanmax(diff_db), np.nanmean(diff_db))

    
    water_thresh = WATER_THRESH_DB
    flood_raw = (pre_filt_db > water_thresh) & (post_filt_db < water_thresh)
    water_raw = (pre_filt_db < water_thresh) & (post_filt_db < water_thresh)

    flood_raw_path = os.path.join(OUTPUT_DIR, "flood_raw_mask.tif")
    water_raw_path = os.path.join(OUTPUT_DIR, "water_raw_mask.tif")
    save_uint8_mask(flood_raw, meta, flood_raw_path)
    save_uint8_mask(water_raw, meta, water_raw_path)
    print("Saved raw masks:", flood_raw_path, water_raw_path)
    print("Raw counts - flood:", int(np.sum(flood_raw)), " water:", int(np.sum(water_raw)))

    
    if np.sum(flood_raw) == 0 and np.sum(water_raw) == 0:
        print("No raw pixels found. Trying fallback thresholds...")
        found = False
        for wt in [-22, -20, -18, -16]:
            ftmp = (pre_filt_db > wt) & (post_filt_db < wt)
            wtmp = (pre_filt_db < wt) & (post_filt_db < wt)
            if np.sum(ftmp) >= MIN_SIZE_RAW or np.sum(wtmp) >= MIN_SIZE_RAW:
                print("Fallback found wt:", wt)
                flood_raw = ftmp; water_raw = wtmp; found = True; break
        if not found:
            print("Fallback did not find features. Check band index, AOI, or preprocessing.")

    
    def clean_mask(raw_mask, opening_radius=2, min_size=MIN_SIZE_CLEAN):
        cleaned = median_filter(raw_mask.astype(np.uint8), size=3).astype(bool)
        cleaned = binary_opening(cleaned, footprint=disk(opening_radius))
        cleaned = remove_small_objects(cleaned, min_size=min_size)
        if np.sum(cleaned) == 0 and np.sum(raw_mask) > 0:
            print("Cleaning removed everything: keeping raw mask (less aggressive)")
            return raw_mask
        return cleaned

    flood_clean = clean_mask(flood_raw)
    water_clean = clean_mask(water_raw)

    flood_mask_path = os.path.join(OUTPUT_DIR, "flood_mask.tif")
    water_mask_path = os.path.join(OUTPUT_DIR, "water_mask.tif")
    save_uint8_mask(flood_clean, meta, flood_mask_path)
    save_uint8_mask(water_clean, meta, water_mask_path)
    print("Saved cleaned masks:", flood_mask_path, water_mask_path)

    # area calculation (hectares)
    pixel_width = meta['transform'][0]*100000
    pixel_height = meta['transform'][4]*100000 
    pixel_area_m2 = abs(pixel_width * pixel_height)
    pixel_area_ha = pixel_area_m2 / 10000.0

    # flooded area = post-flood mask
    flood_area_pixels = int(np.sum(flood_clean))
    flood_area_ha = flood_area_pixels * pixel_area_ha

    # water area = pre-flood mask
    water_area_pixels = int(np.sum(water_clean))
    water_area_ha = water_area_pixels * pixel_area_ha

    print(f"Flooded Area (post-flood): {flood_area_ha:.2f} ha (pixels: {flood_area_pixels})")
    print(f"Water Area (pre-flood): {water_area_ha:.2f} ha (pixels: {water_area_pixels})")

    # Flatten arrays
    y_true = water_clean.astype(int).flatten()  # Pre-flood water as proxy "truth"
    y_pred = flood_clean.astype(int).flatten()  # Post-flood flood mask as "prediction"

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (Pre-flood vs Post-flood):\n", cm)

    # Report
    print("\nClassification Report:\n",
        classification_report(y_true, y_pred, target_names=["No Water", "Water/Flood"]))
    
    # save combined classification
    combined = np.zeros_like(pre_raw, dtype=np.uint8)
    combined[flood_clean] = 1
    combined[water_clean] = 2
    combined_path = os.path.join(OUTPUT_DIR, "flood_water_classified.tif")
    save_uint8_mask(combined, meta, combined_path)
    print("Saved combined classified raster:", combined_path)

    # quick plot
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1); plt.title("Pre VH (dB)"); plt.imshow(pre_filt_db, vmin=-30, vmax=0); plt.colorbar()
    plt.subplot(1,3,2); plt.title("Post VH (dB)"); plt.imshow(post_filt_db, vmin=-30, vmax=0); plt.colorbar()
    plt.subplot(1,3,3); plt.title("Result (yellow=flood, blue=water)")
    rgb = np.zeros((combined.shape[0], combined.shape[1], 3), dtype=float)
    rgb[combined==1] = [1,1,0]; rgb[combined==2] = [0,0,1]
    plt.imshow(rgb)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
