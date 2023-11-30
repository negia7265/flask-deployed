import pytesseract
import re
import shutil
import os
import random
from IPython.display import display
try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import numpy as np
from IPython.display import display, Image as IPImage
import pdfplumber
from flair.data import Sentence
from flair.nn import Classifier
from flair.models import SequenceTagger
# load flask
from flask import Flask, jsonify, request
from flask import Flask
from flask_cors import CORS, cross_origin
from city import cities
# load the NER tagger

app = Flask(__name__)

# tagger = Classifier.load('ner-large')
tagger = SequenceTagger.load('ner-large-model')


@app.route('/process_image', methods=['POST'])
def get_address():
    # Load Model
    # tagger = Classifier.load('ner-large')
    # Load PDF or Image

    if 'file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400
    extractedInformation = ""
    if file and file.filename.endswith('.pdf'):
        num_pages_to_extract = 2
        text = ''
        with pdfplumber.open(file) as pdf:
            for page_num in range(min(num_pages_to_extract, len(pdf.pages))):
                page = pdf.pages[page_num]
                page = page.dedupe_chars(tolerance=1)
                page_text = page.extract_text()
                text += page_text + '\n'
                extractedInformation = text
    elif file.filename.endswith(('.jpg', '.jpeg', '.png', '.PNG')):
        print("Image received")
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_pillow = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        extractedInformation = pytesseract.image_to_string(img_pillow)
    else:
        print('Invalid file format. Please upload a PDF file.')

    sentence = Sentence(extractedInformation)

    # run NER over sentence
    tagger.predict(sentence)

    # print the sentence with all annotations
    # print(sentence)

    # Extract tokens labeled as 'LOC' from the named entities
    loc_tokens = [str(entity) for entity in sentence.get_spans(
        'ner') if 'LOC' in str(entity.labels[0])]

    # Print the list of 'LOC' tokens
    loc_texts_inside_quotes = [
        re.search(r'"([^"]*)"', loc_token).group(1) for loc_token in loc_tokens]

    # # Print the list of texts inside double quotes
    # print(loc_texts_inside_quotes)

    # cities = ["Kolkata", "Bengaluru", "Dehradun",
    #           "Chennai", "Mumbai", "Alwar", "Jaipur", "Pune"]
    states = ["Rajasthan", "Karnataka", "Uttarakhand"]

    def IndiaExist(temp):
        for i in range(len(temp)):
            if (temp[i] == "India"):
                return True

    def CityExist(city):
        for i in range(len(cities)):
            if (cities[i] == city):
                return True

    def StateExist(state):
        for i in range(len(states)):
            if (states[i] == state):
                return True

    list_again = []
    if (IndiaExist(loc_texts_inside_quotes)):
        for i in range(len(loc_texts_inside_quotes)):
            if (loc_texts_inside_quotes[i] == "India"):
                list_again.append(i)
    else:
        for i in range(len(loc_texts_inside_quotes)):
            if (CityExist(loc_texts_inside_quotes[i])):
                list_again.append(i)

    if len(list_again) >= 1:
        # Extract the substring from the beginning to the first occurrence of "India"
        substring_before_first_india = loc_texts_inside_quotes[:list_again[0] + 1]

        # Extract the substring from the first occurrence of "India" to the second occurrence of "India"
        if len(list_again) >= 2:
            substring_between_indias = loc_texts_inside_quotes[list_again[0] + 1:list_again[1]+1]
        else:
            substring_between_indias = loc_texts_inside_quotes[list_again[0] + 1:]

        # substring_between_indias = corrected_addresses[list_again[0] + 1:list_again[1] + 1]
    # removing white spaces

    cleaned_first_address = [word.strip()
                             for word in substring_before_first_india if word.strip()]
    cleaned_second_address = [word.strip()
                              for word in substring_between_indias if word.strip()]

    first_address = ', '.join(cleaned_first_address)
    second_address = ', '.join(cleaned_second_address)

    # Print the combined string
    return {'status': "File Uploaded", 'source': first_address, 'destination': second_address}


app.run()
