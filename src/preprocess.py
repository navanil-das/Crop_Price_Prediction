import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    df = df.dropna()

    return df


def prepare_features(df):

    X = df[['Min_x0020_Price','Max_x0020_Price']]
    y = df['Modal_x0020_Price']

    return X, y
