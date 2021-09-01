import os
import numpy as np
import imageio
from PIL import Image

def human_detect(input_frame):

    return None

def video_read(link_video):
    frame_reader = imageio.get_reader(link_video, 'ffmpeg', mode='I')
    idx, pre_idx = 0, 0
    for frame_num, frame in enumerate(frame_reader):
        face_frame, bbox = human_detect(frame)
        im = Image.fromarray(face_frame.astype('uint8'))
        im.save('{}_human.png'.format(link_video.replace('.mp4', '')))

    return None
def main():
    pass
if __name__ == "__main__":
    main()