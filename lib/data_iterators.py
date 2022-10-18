import os
import cv2
import glob

def is_a_file(target_file, exts):
    """Check if exts contain the type of the target_file"""
    if not os.path.isfile(target_file):
        return False
    if any([target_file.lower().endswith(ext) for ext in exts]):
        return True
    return False

class VideoReader:
    """An iterator for a video"""
    def __init__(self, video_path):
        self.vidcap = cv2.VideoCapture(video_path)
        self.vid_fps = self.vidcap.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        self.count = 0
        return self
            
    def __next__(self):
        frame_exists, curr_frame = self.vidcap.read()
        if frame_exists:
            self.count += 1
            return cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        elif self.count > 0:
            raise StopIteration

    def get_fps(self):
        return self.vid_fps

class ImageReader:
    """An iterator to iterate the images of a folder in ascending order

    It is adaptable when file_path is a single image

    """
    def __init__(self, file_path, img_types=('.png', '.jpg'), fps=None):
        self.img_paths = []
        if is_a_file(file_path, img_types):
            self.img_paths.append(file_path)
        else:
            assert os.path.isdir(file_path)

            # case sensitive to img_types
            for img_type in img_types:
                self.img_paths.extend(
                        glob.glob(os.path.join(file_path, '*' + img_type))
                        )
            self.img_paths = sorted(self.img_paths)
        self.fps = fps

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.img_paths):
            img = cv2.imread(self.img_paths[self.idx])
            self.idx += 1
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise StopIteration

    def get_fps(self):
        return self.fps

class SeqReader:
    """An iterator to iterate the images of a folder in ascending order

    It is adaptable when file_path is a single image

    """
    def __init__(self, file_path, img_types=('.png', '.jpg'), fps=None):
        self.img_paths = []
        self.file_path = file_path

        if is_a_file(file_path, img_types):
            self.img_paths.append(file_path)
        else:
            assert os.path.isdir(file_path)

            # case sensitive to img_types
            for img_type in img_types:
                self.img_paths.extend(
                        glob.glob(os.path.join(file_path, '*' + img_type))
                        )
            self.img_paths = sorted(self.img_paths)
        self.fps = fps

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.img_paths)-1:
            tgt_img = cv2.imread(self.img_paths[self.idx])
            src_img = cv2.imread(self.img_paths[self.idx+1])
            self.idx += 1
            tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            return (tgt_img.copy(), src_img.copy())
        else:
            raise StopIteration

    def get_fps(self):
        return self.fps
