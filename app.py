from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("forestfiremodel.pkl")


@app.route("/")
def hello_world():
    return render_template("index.html", pred=False)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = "{0:.{1}f}".format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template(
            "index.html",
            pred=True,
            probability=output,
        )
    else:
        return render_template(
            "index.html",
            pred=True,
            probability=output,
        )


if __name__ == "__main__":
    app.run(debug=True, port=80)
