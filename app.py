from flask import Flask, request, render_template

from src.logger.logger import logging
from src.exceptions.exception import CustomException
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=["GET","POST"])
def predict_data_point():
    if request.method=="GET":
        return render_template("form.html")
    else:
        cust_data = CustomData(
            carat = float(request.form.get("carat")),
            depth = float(request.form.get("depth")),
            table = float(request.form.get("table")),
            x = float(request.form.get("x")),
            y = float(request.form.get("y")),
            z = float(request.form.get("z")),
            cut = request.form.get("cut"),
            color = request.form.get("color"),
            clarity = request.form.get("clarity"),
        )
        test_data = cust_data.arr_to_datafarme()

        pred_pipeline = PredictionPipeline()
        preds = pred_pipeline.predict(test_data)

        result = round(preds[0],2)
        return render_template("results.html",result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)