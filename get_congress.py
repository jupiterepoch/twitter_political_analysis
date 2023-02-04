import pandas as pd
import numpy as np
import pdb
from collections import defaultdict

def parse_file():
    df1 = pd.read_json('legislators-current.json')
    df2 = pd.read_json('legislators-social-media.json')
    ids, twt = defaultdict(lambda:[]), defaultdict(lambda:[])

    for i in range(df1.shape[0]):
        try:
            idx = df1["id"][i]["bioguide"]
            bir = df1["bio"][i]["birthday"]
            gdr = df1["bio"][i]["gender"]
            pty = df1["terms"][i][-1]["party"]
            ids["bioguide"].append(idx)
            ids["birthday"].append(bir)
            ids["gender"].append(gdr)
            ids["party"].append(pty)
        except KeyError:
            continue
    
    for i in range(df2.shape[0]):
        try:
            idx = df2["id"][i]["bioguide"]
            twn = df2["social"][i]["twitter"]
            twi = df2["social"][i]["twitter_id"]
            twt["bioguide"].append(idx)
            twt["twitter"].append(twn)
            twt["twitter_id"].append(np.int64(twi))
        except KeyError:
            continue

    df1 = pd.DataFrame(ids).set_index("bioguide")
    df2 = pd.DataFrame(twt).set_index("bioguide")
    df2['twitter_id'] = df2['twitter_id'].astype('Int64')
    df = df1.join(df2)
    df = df.rename(columns={"twitter_id":"uid"})
    print(df.shape[0])

    print(df["uid"].isna().sum())
    df.dropna(inplace=True)
    df['uid'] = df['uid'].astype(np.int64)
    print(df.dtypes)
    df.to_pickle("congress.pkl")


if __name__ == '__main__':
    parse_file()
