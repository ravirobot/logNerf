import cv2

def bgr_to_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def rgb_to_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def display_img_from_path(img_path, window_name='image'):
    img = load_img(img_path)
    display_img(img, window_name=window_name)

def load_img(img_path):
    img = cv2.imread(img_path)
    if img is not None:
        return img
    raise Exception(f'Failed to read {img_path}')

def display_img(img, window_name='image'):
    if img is not None:
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise Exception(f'Invalid image.')
    
def display_rgb_img(img, window_name='image'):
    if img is not None:
        img = rgb_to_bgr(img)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img = bgr_to_rgb(img)
    else:
        raise Exception(f'Invalid image.')

def get_img_max_min_val(img):
        return (img.max(axis=(0, 1)), img.min(axis=(0, 1)))
