from flask import Flask, render_template, request, url_for, session, send_from_directory, Response, jsonify
from camera import VideoCamera
from flask_cors import CORS
import os
import cv2
from keras.preprocessing import image
from preprocess import PreProcess
from predict_caption import PredictCaption
from get_frames import FrameCamera
from threading import Thread

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

ans=''

preprocess_image=PreProcess()
predict_caption=PredictCaption()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/select")
def select_file():
    return render_template("select.html")

@app.route("/live")
def live():
    return render_template("live.html")

# UPLOAD FILE
@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, "uploads")
    print(target)
    file = request.files['file']
    print(file)
    filename = file.filename
    destination = "\\".join([target, filename])
    print(destination)
    file.save(destination)
    session['uploadFilePath']=destination
    record=FrameCamera(destination)
    tot=record.tot_frames()
    print('tot',tot)
    if tot<80:
        response = "Video too small to generate caption"
        return response
    else:
        train_img=record.get_frame()
        print(len(train_img))
        # print(train_img)
        print("extracting features")
        ans=''
        features=preprocess_image.extract_features(train_img,10)

        print("generating caption")
        ans=predict_caption.greedysearch(features)
        print(ans)

        return render_template("output.html", destination=destination, ans=ans)

def predict(camera):
    train_img=[]
    while True:
        frame = camera.get_frame()
        temp_img=image.img_to_array(cv2.resize(frame, dsize=(224, 224)))
        train_img.append(temp_img)

        if len(train_img)==80:
            #extract the features of the arrays from VGG16 model
            print("extracting features")
            ans=''
            features=preprocess_image.extract_features(train_img,10)

            print("generating caption")
            ans=predict_caption.greedysearch(features)
            print(ans)
            train_img=[]




def gen(camera):
    th=Thread(target=predict, args=(camera,))
    th.start()
    while True:
        frame = camera.get_frame()
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    th.join()

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
