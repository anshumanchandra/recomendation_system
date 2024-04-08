import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from flask import session, redirect, url_for
from flask import jsonify
import pandas as pd

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

df = pd.read_csv('styles.csv', engine='python', on_bad_lines='skip')

grouped_products = df.groupby('masterCategory')

def get_random_products_by_category(grouped_products):
    random_products = {}
    for category, products in grouped_products:
        sample_size = min(4, len(products))
        random_products[category] = products.sample(sample_size).to_dict('records')
    return random_products

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

from flask import Flask, url_for, render_template, request

app = Flask(__name__)
#app._static_folder = os.path.join(os.getcwd(), 'show-more/static')
app.secret_key = 'your_secret_key_here'

@app.route('/store_image_path', methods=['POST'])
def store_image_path():
    img_path = request.form.get('img_path')
    session['selected_image_path'] = img_path
    return jsonify({'success': True})

@app.route('/')
def landing_page():
    random_products_by_category = get_random_products_by_category(grouped_products)
    return render_template('landing.html', random_products_by_category=random_products_by_category)

@app.route('/recommend', methods=['GET'])
def recommend_page():
    img_path = session.get('selected_image_path', default='static/images/33634.jpg')
    print(f"Retrieved img_path: {img_path}")
    if img_path is None or not os.path.exists(img_path):
        print("Invalid img_path")
        return redirect(url_for('landing_page'))
    else:
        product_id = os.path.splitext(os.path.basename(img_path))[0]
        product_details = df[df['id'] == int(product_id)].to_dict('records')[0]
        user_features = feature_extraction(img_path, model)
        recommended_indices = recommend(user_features, feature_list)
        recommended_paths = [filenames[i] for i in recommended_indices[0]]
        return render_template('recommend.html', product=product_details, images=recommended_paths)

if __name__ == "__main__":
    app.run(debug=True)