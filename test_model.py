
import joblib
import json
import sklearn

print("Scikit-learn version:", sklearn.__version__)

model = joblib.load("genetic_disorder_model.joblib")
print("Model loaded successfully!")

with open("model_info.json", "r") as file:
    model_info = json.load(file)

print("\nModel Information:")
print(json.dumps(model_info, indent=4))
print("\nExpected input features:")
print(model.feature_names_in_)
