import cv2
from keras.preprocessing import image

class FrameCamera(object):
    def __init__(self,path):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def tot_frames(self):
        self.tot=self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        return self.tot

    def get_frame(self):
        print(self.tot)
        no_frames=81
        n=int(self.tot/no_frames)

        print(n)
        count=0
        c=0
        success=1
        frames=[]
        while success and c<80:
            success, frame = self.video.read()
            if count%n==0:
                c+=1
                # print(count,success)
                temp_img=image.img_to_array(cv2.resize(frame, dsize=(224, 224)))
                # print(temp_img)
                frames.append(temp_img)
            count+=1

        print(len(frames))

        return frames
