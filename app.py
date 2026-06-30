from flask import Flask, render_template, request
import pickle
import sys

app = Flask(__name__)

# ===========================
# Load Model
# ===========================

try:
    with open("Naive_Bayes_Model.pkl", "rb") as f:
        model = pickle.load(f)

except FileNotFoundError:
    model = None
    print("Model file not found.")

except Exception as e:
    model = None
    err_type, err_msg, err_line = sys.exc_info()

    print(
        f"Error from line no : {err_line.tb_lineno} "
        f"due to {err_type.__name__} "
        f"reason was : {err_msg}"
    )


# ===========================
# Home Page
# ===========================

@app.route("/")
def home():

    try:
        return render_template("index.html")

    except Exception as e:

        err_type, err_msg, err_line = sys.exc_info()

        return (
            f"Error from line no : {err_line.tb_lineno} "
            f"due to {err_type.__name__} "
            f"reason was : {err_msg}"
        )


# ===========================
# Prediction
# ===========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:
            return render_template(
                "index.html",
                prediction_text="Model is not loaded."
            )

        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])

        result = model.predict([[sl, sw, pl, pw]])[0]

        if result == 0:
            prediction = "Setosa Flower"

        elif result == 1:
            prediction = "Versicolor Flower"

        else:
            prediction = "Virginica Flower"

        return render_template(
            "index.html",
            prediction_text=prediction
        )

    except ValueError:

        return render_template(
            "index.html",
            prediction_text="Please enter valid numeric values."
        )

    except KeyError:

        return render_template(
            "index.html",
            prediction_text="Missing input field."
        )

    except Exception:

        err_type, err_msg, err_line = sys.exc_info()

        return render_template(
            "index.html",
            prediction_text=(
                f"Error from line no : {err_line.tb_lineno} "
                f"due to {err_type.__name__} "
                f"reason was : {err_msg}"
            )
        )


# ===========================
# Main
# ===========================

if __name__ == "__main__":

    try:
        app.run(debug=True)

    except Exception:

        err_type, err_msg, err_line = sys.exc_info()

        print(
            f"Error from line no : {err_line.tb_lineno} "
            f"due to {err_type.__name__} "
            f"reason was : {err_msg}"
        )