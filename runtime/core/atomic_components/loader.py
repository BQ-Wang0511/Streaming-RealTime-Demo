import filetype
import imageio
import cv2


def is_image(file_path):
    return filetype.is_image(file_path)


def is_video(file_path):
    return filetype.is_video(file_path)


def check_resize(h, w, max_dim=1920, division=2):
    rsz_flag = False
    # ajust the size of the image according to the maximum dimension
    if max_dim > 0 and max(h, w) > max_dim:
        rsz_flag = True
        if h > w:
            new_h = max_dim
            new_w = int(round(w * max_dim / h))
        else:
            new_w = max_dim
            new_h = int(round(h * max_dim / w))
    else:
        new_h = h
        new_w = w

    # ensure that the image dimensions are multiples of n
    if new_h % division != 0:
        new_h = new_h - (new_h % division)
        rsz_flag = True
    if new_w % division != 0:
        new_w = new_w - (new_w % division)
        rsz_flag = True

    return new_h, new_w, rsz_flag

 
def load_image(image_path, max_dim=-1):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    new_h, new_w, rsz_flag = check_resize(h, w, max_dim)
    if rsz_flag:
        img = cv2.resize(img, (new_w, new_h))
    return img


def load_video(video_path, n_frames=-1, max_dim=-1):
    reader = imageio.get_reader(video_path, "ffmpeg")

    new_h, new_w, rsz_flag = None, None, None

    ret = []
    for idx, frame_rgb in enumerate(reader):
        if n_frames > 0 and idx >= n_frames:
            break

        if rsz_flag is None:
            h, w = frame_rgb.shape[:2]
            new_h, new_w, rsz_flag = check_resize(h, w, max_dim)

        if rsz_flag:
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        ret.append(frame_rgb)

    reader.close()
    return ret


def load_source_frames(source_path, max_dim=-1, n_frames=-1):
    if is_image(source_path):
        rgb = load_image(source_path, max_dim)
        rgb_list = [rgb]
        is_image_flag = True
    elif is_video(source_path):
        rgb_list = load_video(source_path, n_frames, max_dim)
        is_image_flag = False
    else:
        raise ValueError(f"Unsupported source type: {source_path}")
    return rgb_list, is_image_flag


def _mirror_index(index, size):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1
    

class LoopLoader:
    def __init__(self, item_list, max_iter_num=-1, mirror_loop=True):
        self.item_list = item_list
        self.idx = 0
        self.item_num = len(self.item_list)
        self.max_iter_num = max_iter_num if max_iter_num > 0 else self.item_num
        self.mirror_loop = mirror_loop

    def __len__(self):
        return self.max_iter_num
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx >= self.max_iter_num:
            raise StopIteration
        
        if self.mirror_loop:
            idx = _mirror_index(self.idx, self.item_num)
        else:
            idx = self.idx % self.item_num
        item = self.item_list[idx]

        self.idx += 1
        return item
    
    def __call__(self):
        return self.__iter__()
    
    def reset(self, max_iter_num=-1):
        self.frame_idx = 0
        self.max_iter_num = max_iter_num if max_iter_num > 0 else self.item_num
    





