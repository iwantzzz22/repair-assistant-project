
from flask import Flask, request, jsonify
import asyncio
import io
import base64
from google.cloud import vision
from google.cloud import language_v1
import firebase_admin
from firebase_admin import credentials, db

# Initialize Flask app
app = Flask(__name__)

# Set Google Cloud Vision and Firebase credentials paths
GOOGLE_CRED_PATH = 'cac2024-437923-f03f6eaf6b3a.json'
FIREBASE_CRED_PATH = 'cac2024-23ede-firebase-adminsdk-u9z22-1e58186f66.json'
FIREBASE_URL = 'https://cac2024-23ede-default-rtdb.firebaseio.com/'

def initialize_google_credentials():
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_CRED_PATH

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})

# Image recognition function using Google Cloud Vision API
async def detect_labels(image_data):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)
    response = await asyncio.to_thread(client.label_detection, image=image)
    labels = [label.description for label in response.label_annotations]
    return labels

# Text analysis function using Google Cloud Natural Language API
async def analyze_text(text_content):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = await asyncio.to_thread(client.analyze_entities, document=document)
    entities = {entity.name: entity.type_ for entity in response.entities}
    return entities

# Fetch repair suggestions from Firebase
async def get_repair_suggestions(problem_type):
    ref = db.reference(f'repairs/{problem_type}')
    suggestions = await asyncio.to_thread(ref.get)
    return suggestions

# Flask route to handle Thunkable requests
@app.route('/analyze', methods=['POST'])
async def analyze():
    data = request.json
    image_base64 = data['image_data']
    text_content = data['text_content']

    # Decode base64 image data to bytes
    image_data = base64.b64decode(image_base64.split(",")[1])

    labels = await detect_labels(image_data)
    entities = await analyze_text(text_content)
    problem_type = next(iter(entities), 'general')

    initialize_firebase()
    suggestions = await get_repair_suggestions(problem_type)

    return jsonify({'labels': labels, 'entities': entities, 'suggestions': suggestions})

if __name__ == '__main__':
    initialize_google_credentials()
    app.run(host='0.0.0.0', port=5000)
