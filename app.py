from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import ollama
import concurrent.futures  # 🚀 Multi-threading for faster execution

app = Flask(__name__)

# Load the trained model once at startup
model = tf.keras.models.load_model(r"C:\Users\andre\Simple-Food-Image-Classifier\Dataset1.5kClass\1.5kClass_32x32\weights\weights.67-0.9205-0.2702.h5")

# Define class labels
class_labels = ["Sushi", "Mushroom", "Onion", "Rice"]  # Modify as per your dataset

def generate_llm_response(prompt):
    """Generate a response from Ollama."""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    text = response['message']['content']
    return "\n".join(f"• {point.strip()}" for point in text.split("\n")[:10] if point.strip())

def predict_image(image):
    """Classify image and get food details."""
    img = Image.open(io.BytesIO(image)).resize((32, 32))  # Resize for faster processing
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[0][class_index])
    food_name = class_labels[class_index]

    # Use Multi-threading to speed up LLM responses 🚀
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_desc = executor.submit(generate_llm_response, f"Provide a detailed description of {food_name} in 10 bullet points.")
        future_nutrition = executor.submit(generate_llm_response, f"Provide nutritional information for {food_name} in 10 bullet points.")
        future_recipe = executor.submit(generate_llm_response, f"Provide a step-by-step recipe for {food_name} in 10 bullet points.")

        food_description = future_desc.result()
        nutrition_info = future_nutrition.result()
        food_recipe = future_recipe.result()

    return {
        "class": food_name,
        "confidence": confidence,
        "description": food_description,
        "nutrition": nutrition_info,
        "recipe": food_recipe
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file'].read()
    result = predict_image(file)
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Removed debug=True for faster execution
