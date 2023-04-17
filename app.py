from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
 0:":Apple___Apple_scab",
 1:'Apple___Black_rot',
 2:'Apple___Cedar_apple_rust',
 3:'Apple___healthy',
 4:'Blueberry___healthy',
 5:'Cherry_(including_sour)___Powdery_mildew',
 6:'Cherry_(including_sour)___healthy',
 7:'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8:'Corn_(maize)___Common_rust_',
 9:'Corn_(maize)___Northern_Leaf_Blight',
 10:'Corn_(maize)___healthy',
 11:'Grape___Black_rot',
 12:'Grape___Esca_(Black_Measles)',
 13:'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14:'Grape___healthy',
 15:'Orange___Haunglongbing_(Citrus_greening)',
 16:'Peach___Bacterial_spot',
 17:'Peach___healthy',
 18:'Pepper,_bell___Bacterial_spot',
 19:'Pepper,_bell___healthy',
 20:'Potato___Early_blight',
 21:'Potato___Late_blight',
 22:'Potato___healthy',
 23:'Raspberry___healthy',
 24:'Soybean___healthy',
 25:'Squash___Powdery_mildew',
 26:'Strawberry___Leaf_scorch',
 27:'Strawberry___healthy',
 28:'Tomato___Bacterial_spot',
 29:'Tomato___Early_blight',
 30:'Tomato___Late_blight',
 31:'Tomato___Leaf_Mold',
 32:'Tomato___Septoria_leaf_spot',
 33:'Tomato___Spider_mites Two-spotted_spider_mite',
 34:'Tomato___Target_Spot',
 35:'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36:'Tomato___Tomato_mosaic_virus',
 37:'Tomato___healthy'}



model = load_model('plants.h5')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name[classes_x[0]]

 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/performance")
def performance():
	return render_template('performance.html')
    
@app.route("/chart")
def chart():
	return render_template('chart.html') 

	
if __name__ =='__main__':
	app.run(debug = True)


	

	


