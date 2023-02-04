import pandas as pd
def save_data(data, name):
    data = pd.DataFrame(data)
    data.to_pickle(f"{name}.pkl")