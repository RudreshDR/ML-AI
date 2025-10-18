'''
    #small eg: how it work..
    X_train = [[1],[2],[3]]
    y_train = [3,5,7]
    # Training data
    model = LinearRegression()
    print("model",model)
    model.fit(X_train, y_train)  # learns y = 2*x + 1
    X_test = [[4],[5]]
    y_pred = model.predict(X_test)
    print("y_pred",y_pred) #--->[ 9. 11.]
'''
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from data_loader import load_and_prepare_data #it wil take data from out src folder dataloader script
import os
os.makedirs("model", exist_ok=True)
path = 'salary_prediction_ml_mini_project\data\salary_data.csv'
def train_model():
    x_train,x_test,y_train,y_test=load_and_prepare_data(path)
    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    print("model trained sucessfully")
    print(f"r2_score is {r2_score(y_test,y_pred):.3f}")
    print(f"RMSE is {mean_squared_error(y_test,y_pred):.2f}")
    joblib.dump(model, "model/salary_predictor.pkl")
    print("Model saved at model/salary_predictor.pkl")

if __name__=='__main__':
    train_model()