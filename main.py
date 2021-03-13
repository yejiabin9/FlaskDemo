from datetime import datetime
from flask import Flask, request, jsonify, render_template
import os
import random
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# # 获取当前位置的绝对路径
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG'])
covid_pneumo_model = load_model('./model/covid_normal_pneumonia_model.h5')


def test_rx_image_for_Covid19(model, imagePath, filename):
    img = cv2.imread(imagePath)
    img_out = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = model.predict(img)
    print(pred)
    pred_mid = int((pred[0][0] * 100))
    pred_neg = int((pred[0][1] * 100))
    pred_pos = int((pred[0][2] * 100))

    if np.argmax(pred, axis=1)[0] == 0:
        prediction = 'PNEUMONIA'
        prob = pred_mid
    if np.argmax(pred, axis=1)[0] == 1:
        prediction = 'NEGATIVE'
        prob = pred_neg
    if np.argmax(pred, axis=1)[0] == 2:
        prediction = 'POSITIVE'
        prob = pred_pos

    img_pred_name = prediction + '_Prob_' + str(prob) + '_Name_' + filename + '.png'
    cv2.imwrite('static/xray_analisys/' + img_pred_name, img_out)
    cv2.imwrite('static/Image_Prediction.png', img_out)
    print
    return prediction, prob


def img_compare(path):
    img_1 = cv2.imread("./static/Image_Prediction.png")
    img1 = cv2.resize(img_1, (224, 224))
    img_2 = cv2.imread(path)
    img2 = cv2.resize(img_2, (224, 224))
    res = np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2))
    return res


# 上传照片test接口
@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get('file')
    # 获取安全的文件名 正常的文件名
    filename = secure_filename(f.filename)
    print(filename)

    # 生成随机数
    random_num = random.randint(0, 100)

    # f.filename.rsplit('.', 1)[1] 获取文件的后缀
    # 把文件重命名
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random_num) + "." + filename.rsplit('.', 1)[1]

    # 保存的目标绝对地址
    file_path = basedir + "/static/file/"

    # 判断文件夹是否存在 不存在则创建
    if not os.path.exists(file_path):
        os.makedirs(file_path, 755)

    # 保存文件到目标文件夹
    f.save(file_path + filename)
    print(file_path)
    print(filename)
    res = img_compare(file_path+filename)
    print(res)
    if res<0.4:
        prediction = "Raken"
        prob = "Raken"
    else:

        prediction, prob = test_rx_image_for_Covid19(covid_pneumo_model, file_path + filename, filename)
    print(prediction)
    print(prob)
    return render_template('index.html', prediction=prediction, confidence=prob, filename=filename,
                           file_path="file/" + filename)

    # 返回前端可调用的一个链接
    # 可以配置成对应的外网访问的链接
    # my_host = "http://127.0.0.1:5000"
    # new_path_file = my_host + "/static/file/" + filename
    # data = {"prediction":prediction,"prob":prob}
    #
    # payload = jsonify(data)
    # return payload, 200


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 启动初始化HTML
@app.route('/')
def hello_world():
    return render_template('index.html', )


@app.route("/query", methods=["POST"])
def query():
    if request.method == 'POST':
        # RECIBIR DATA DEL POST
        if 'file' not in request.files:
            return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        file = request.files['file']
        # image_data = file.read()
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        if file and allowed_file(file.filename):

            filename = str(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            image_name = filename

            # detection covid
            try:
                prediction, prob = test_rx_image_for_Covid19(covid_pneumo_model, img_path, filename)
                return render_template('index.html', prediction=prediction, confidence=prob, filename=image_name,
                                       xray_image=img_path)
            except:
                return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename=image_name,
                                       xray_image=img_path)
        else:
            return render_template('index.html', name='FILE NOT ALOWED', confidence=0, filename=image_name,
                                   xray_image=img_path)


# No caching at all for API endpoints.

# @app.after_request
# def add_header(response):
#     response.headers['Cache-Control'] = 'public, max-age=0'
#     return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
