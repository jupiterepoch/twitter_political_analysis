#!/usr/bin/env python3
"""Collect tweets from Twitter streaming API via tweepy"""

import argparse
import datetime
import gzip
import os
import sys
import time
import pandas as pd
import numpy as np
from tweepy import Stream, Client, StreamingClient, StreamRule, Paginator
from pprint import pprint
import pickle
import json
import pdb
from collections import defaultdict
from tqdm import tqdm


def eprint(*args, **kwargs):
    """Print to stderr"""
    print(*args, file=sys.stderr, **kwargs)


def save_data(data, name):
    data = pd.DataFrame(data)
    data.to_pickle(f"{name}.pkl")


if __name__ == "__main__":

    # Read twitter app credentials and set up authentication
    creds = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds2.txt", "r").readlines() if line}

    # Read macros
    congress = pd.read_pickle("congress.pkl")
    user_ids = congress["uid"].to_numpy().tolist()

    # Track time and start streaming
    twitter_client = Client(bearer_token=creds["bearer_token"], wait_on_rate_limit=True)

    # all conversations started by congress, identified by uid
    conversations = {
        "uid": [],
        "tweet_id": [],
    }
    for idx, uid in enumerate(tqdm(user_ids)):
        args = {
            'id': uid, 'exclude': ["retweets", "replies"],
            'tweet_fields': ["id"], 'start_time': "2022-10-29T12:00:00Z"
        }
        try:
        # all the conversation info will be found later
            recent_tweets = Paginator(
                twitter_client.get_users_tweets,
                **args    
            ).flatten()
            
            for i, t in enumerate(recent_tweets):
                tweet_id = np.int64(t.id) # twitter_id is thread conversation_id     
                conversations["uid"].append(uid)
                conversations["tweet_id"].append(tweet_id)
                    
        except Exception as e:
            eprint(e)
        
        if idx % 3 == 1:
            save_data(conversations, 'conversations_fast')
            
    save_data(conversations, 'conversations_fast')

    