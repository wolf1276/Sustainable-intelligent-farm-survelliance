import io
from google.colab import drive
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

# Mount Google Drive
drive.mount("/content/drive")
# Load the dataset
data_path = '/content/drive/My Drive/Crop_recommendation.csv'
data = pd.read_csv(data_path)

# Assuming the dataset contains relevant columns like 'N', 'P', 'K', 'temperature', 'rainfall', 'humidity', and 'label'
X = data[['N', 'P', 'K','temperature', 'rainfall', 'humidity']]
y = data['label']

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y)

# Crop recommendation function
def recommend_crop(N,P,K,temperature, rainfall, humidity):
    input_data = pd.DataFrame([[N, P, K, temperature, rainfall, humidity]])
    predicted_crop = model.predict(input_data)
    return predicted_crop[0]

# User input for temperature, rainfall, and humidity
N=  int(input("Enter Nitrogen Value:"))
P= int (input("Enter Phosphorus Value:"))
K = int (input( "Enter Potassium Value:"))
temperature = float(input("Enter temperature (in °C): "))
rainfall = float(input("Enter rainfall (in mm): "))
humidity = float(input("Enter humidity (in %): "))

# Recommend crop based on user input
recommended_crop = recommend_crop(N, P, K, temperature, rainfall, humidity)
print("Recommended Crop:", recommended_crop)
