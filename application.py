from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import os

from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

# Define the base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(gender = request.form.get('gender'),
                            race_ethnicity = request.form.get('ethnicity'),
                            parental_level_of_education=request.form.get('parental_level_of_education'),
                            lunch = request.form.get('lunch'),
                            test_preparation_course = request.form.get('test_preparation_course'),
                            reading_score = request.form.get('reading_score'),
                            writing_score = request.form.get('writing_score'))
            
            df = data.get_data_as_dataframe()
            logging.info("data has {}".format(df))
            predict_pipe=PredictPipeline(base_dir=BASE_DIR)
            results=predict_pipe.predict(df)
            logging.info("result is {}".format(results))
            return render_template('home.html',results=results[0])
        except Exception as e:
            logging.error(f"Error occurred in /predictdata: {str(e)}")
            return render_template('home.html', error="An error occurred while processing your request. Please try again.")
            

if __name__ == "__main__":
    app.run(host="0.0.0.0"git, port=8000, debug=True)

