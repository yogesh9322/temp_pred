import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("temp_year.csv")

# Encode categorical variables
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_enc'] = le_country.fit_transform(df['Country'])
df['City_enc'] = le_city.fit_transform(df['City'])

# Features and target
X = df[['Country_enc', 'City_enc', 'Year']]
y = df['AvgTemperature']

# Model training
model = LinearRegression()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(le_country, "le_country.pkl")
joblib.dump(le_city, "le_city.pkl")
