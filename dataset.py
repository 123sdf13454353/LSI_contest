

import numpy as np
from tqdm import tqdm

from config import IMAGE_ROOT, IMAGE_SIZE,Fxp,cv2,glob,os

fxp_ref = Fxp(None, dtype='fxp-s16/8')  # Định nghĩa fixed-point

def get_subset_fixed_point(pathname, name=""):
    images = []
    for fn in tqdm(glob.glob(pathname), desc=name):
        image = cv2.imread(fn, flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE).astype(np.float16) / 255.0
        image_fixed_point = Fxp(image, like=fxp_ref)
        images.append(image_fixed_point)
    return np.array(images)

def load_dataset():
    x_train = get_subset_fixed_point(os.path.join(IMAGE_ROOT, 'train', 'good', '*.png'), 'Train images')
    x_good = get_subset_fixed_point(os.path.join(IMAGE_ROOT, 'test', 'good', '*.png'), 'Good')
    x_crack = get_subset_fixed_point(os.path.join(IMAGE_ROOT, 'test', 'crack', '*.png'), 'Crack')
    x_cut = get_subset_fixed_point(os.path.join(IMAGE_ROOT, 'test', 'cut', '*.png'), 'Cut')
    x_hole = get_subset_fixed_point(os.path.join(IMAGE_ROOT, 'test', 'hole', '*.png'), 'Hole')
    x_print = get_subset_fixed_point(os.path.join(IMAGE_ROOT, 'test', 'print', '*.png'), 'Print')

    x_all = np.vstack((x_train, x_good, x_crack, x_cut, x_hole, x_print))
    np.random.shuffle(x_all)

    split_ratio = 0.8
    split_idx = int(len(x_all) * split_ratio)

    x_train_new = x_all[:split_idx]
    x_test = x_all[split_idx:]

    return x_train_new, x_test,x_good,x_cut,x_train,x_hole,x_print
