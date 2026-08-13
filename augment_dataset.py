import os
import cv2
import numpy as np
import glob
import shutil

# Paths
DATASET_DIR = 'coin_dataset'
IMG_DIR = os.path.join(DATASET_DIR, 'images')
LBL_DIR = os.path.join(DATASET_DIR, 'labels')

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def change_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = 0 - value
        v[v < lim] = 0
        v[v >= lim] -= abs(value)
        
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def flip_horizontal(image, bboxes):
    """
    Flips image and YOLO bounding boxes horizontally.
    YOLO format: class x_center y_center width height
    """
    flipped_img = cv2.flip(image, 1)
    flipped_bboxes = []
    
    for bbox in bboxes:
        # x_center becomes 1.0 - x_center
        cls, x, y, w, h = bbox
        flipped_x = 1.0 - float(x)
        flipped_bboxes.append(f"{cls} {flipped_x:.6f} {y} {w} {h}")
        
    return flipped_img, flipped_bboxes

def run_augmentation():
    print("Starting Data Augmentation...")
    
    # Get all image files in all subdirectories of images/
    image_files = glob.glob(os.path.join(IMG_DIR, '**', '*.jpg'), recursive=True) + \
                  glob.glob(os.path.join(IMG_DIR, '**', '*.png'), recursive=True)
                  
    if not image_files:
        print("No images found to augment.")
        return

    count = 0
    for img_path in image_files:
        # Find corresponding label file
        rel_path = os.path.relpath(img_path, IMG_DIR)
        base_name = os.path.splitext(rel_path)[0]
        
        lbl_path = os.path.join(LBL_DIR, base_name + '.txt')
        
        if not os.path.exists(lbl_path):
            continue
            
        img = cv2.imread(img_path)
        with open(lbl_path, 'r') as f:
            bboxes = [line.strip().split() for line in f.readlines()]
            
        # 1. Flip Horizontal
        f_img, f_boxes = flip_horizontal(img, bboxes)
        f_img_path = img_path.replace('.jpg', '_flip.jpg').replace('.png', '_flip.png')
        f_lbl_path = lbl_path.replace('.txt', '_flip.txt')
        
        cv2.imwrite(f_img_path, f_img)
        with open(f_lbl_path, 'w') as f:
            f.write('\n'.join(f_boxes))
            
        # 2. Brightness (Darker)
        d_img = change_brightness(img, -40)
        d_img_path = img_path.replace('.jpg', '_dark.jpg').replace('.png', '_dark.png')
        d_lbl_path = lbl_path.replace('.txt', '_dark.txt')
        
        cv2.imwrite(d_img_path, d_img)
        shutil.copy(lbl_path, d_lbl_path) # Bboxes don't change
        
        # 3. Brightness (Brighter)
        b_img = change_brightness(img, 40)
        b_img_path = img_path.replace('.jpg', '_bright.jpg').replace('.png', '_bright.png')
        b_lbl_path = lbl_path.replace('.txt', '_bright.txt')
        
        cv2.imwrite(b_img_path, b_img)
        shutil.copy(lbl_path, b_lbl_path)
        
        count += 3
        print(f"Augmented: {base_name} (+3 variations)")

    print(f"Data Augmentation Complete. Generated {count} new images.")

if __name__ == '__main__':
    run_augmentation()
