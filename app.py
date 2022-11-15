from flask import Flask,render_template,request,jsonify
import pickle 
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('picklefile.pkl','rb'))

@app.route("/")
def hello():
    return render_template("home.html")


@app.route("/submit1",methods=["POST"])
def submit1():
    data=request.json["data"]  # type: ignore
    data=np.array(list(data.values())).reshape(1,-1)
    pred=model.predict(data)
    output=pred[0]
    print(output)
    return jsonify(output)

@app.route("/predict",methods=["POST"])  # type: ignore
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    op=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="THe predicted chance is {}".format(op))

if __name__=="__main__":
    app.run(debug=True)