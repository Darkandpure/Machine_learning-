from flask import Flask, request, render_template
from joblib import load
import pandas as pd


app = Flask(__name__)


# Load the Random Forest model
model = load("random_forest_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file:
        # Read the CSV file
        df = pd.read_csv(file)

        # Make predictions
        predictions = model.predict(df)

        # Convert predictions to a Python list
        predictions_list = predictions.tolist()

        # Process data for Hour vs MGD graph
        hour_vs_mgd = {
            'hour': df['HOUR'].tolist(),
            'mgd': predictions_list
        }

        # Extract Month and process data for Month vs MGD graph
        # Assuming 'Date' column is in the format 'YYYY-MM-DD'
        # df['MONTH'] = pd.to_datetime(df['DAY_OF_MONTH']).dt.month
        month_vs_mgd = {
            'month': df['MONTH'].tolist(),
            'mgd': predictions_list
        }

        # Prepare the graph data
        graph_data = {
            'hour_vs_mgd': hour_vs_mgd,
            'month_vs_mgd': month_vs_mgd
        }

        # Render the result template with the predictions and graph data
        return render_template('result.html', predictions=predictions_list, graph_data=graph_data, model_name = "Random Forest", r2_score = "91")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

