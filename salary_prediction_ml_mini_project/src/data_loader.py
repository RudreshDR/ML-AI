import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(path : str):
    df = pd.read_csv(path)
    print("loaded dataset with shape:",df.shape)

    # encode categorical column
    le = LabelEncoder()
    df['education_level'] = le.fit_transform(df['education_level'])
    # Fix salary column (remove commas, convert to float)
    df['salary'] = df['salary'].astype(str).str.replace(',', '').astype(float)

    #split data
    x = df.drop('salary',axis=1)
    y = df['salary']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    #data prepared and splited sucessfully
    return x_train,x_test,y_train,y_test