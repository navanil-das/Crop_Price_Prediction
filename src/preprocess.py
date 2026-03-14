import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True)

    df = df.dropna()

    return df


def prepare_features(df):

    
    df_model = df[[
        "State",
        "Market",
        "Commodity",
        "Min_x0020_Price",
        "Max_x0020_Price",
        "Modal_x0020_Price"
    ]]

    # One-hot encode categorical variables
    df_model = pd.get_dummies(
        df_model,
        columns=["State", "Market", "Commodity"],
        drop_first=True
    )

    X = df_model.drop("Modal_x0020_Price", axis=1)

    y = df_model["Modal_x0020_Price"]

    return X, y
