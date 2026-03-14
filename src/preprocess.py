import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    df = df.dropna()

    return df


def prepare_features(df):

    X = df[['Min Price','Max Price']]
    y = df['Modal Price']

    return X, y
