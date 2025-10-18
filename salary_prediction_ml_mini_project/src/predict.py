import joblib
import pandas as pd

def predict_salary(experience,age,education_level):
    model = joblib.load('model/salary_predictor.pkl')
    edu_map = {"Bachular":0, "Master": 1, "PhD": 2}#LabelEncoder() assignlabel alpabecal order
    data = pd.DataFrame([{
        "experience":experience,
        "age":age,
        "education_level":edu_map.get(education_level, 0)
    }])
    prediction = model.predict(data)[0]
    print(f"Predicted Salary: ₹{prediction:.2f}")


if __name__ == "__main__":
    predict_salary(5, 30, "PhD")