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
from utils import *

def eprint(*args, **kwargs):
    """Print to stderr"""
    print(*args, file=sys.stderr, **kwargs)


if __name__ == "__main__":

    # Read twitter app credentials and set up authentication
    creds = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds.txt", "r").readlines() if line}
    creds1 = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds1.txt", "r").readlines() if line}
    creds2 = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds2.txt", "r").readlines() if line}
    creds3 = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds3.txt", "r").readlines() if line}

    # Read macros
    conv = pd.read_pickle("conversations_fast.pkl")
    conv_ids = conv["tweet_id"].to_numpy()
    np.random.seed(123)
    np.random.shuffle(conv_ids)
    conv_ids = conv_ids.tolist()

    # Track time and start streaming
    client1 = Client(bearer_token=creds["bearer_token"], wait_on_rate_limit=True)
    client2 = Client(bearer_token=creds1["bearer_token"], wait_on_rate_limit=True)
    client3 = Client(bearer_token=creds2["bearer_token"], wait_on_rate_limit=True)
    client4 = Client(bearer_token=creds3["bearer_token"], wait_on_rate_limit=True)
    
    clients = [client1, client2, client3, client4]
    which = int(sys.argv[1])
    client = clients[which]
    
    # all tweets in a conversation, identified by conv_id
    threads = {
        "conv_id": [],
        "uid": [],
        "tweet_id": [],
        "reply_setting": [],
        "parent_tweet_id": [],
    }


    for idx, conv_id in enumerate(tqdm(conv_ids[:len(conv_ids)])):
        if idx % 4 != which:
            continue
        try:
            replies = Paginator(
                
                client.search_recent_tweets,
                query=f'conversation_id:{conv_id}',
                expansions=['referenced_tweets.id'], #  'referenced_tweets.id.author_id'
                tweet_fields=["id", "author_id", "reply_settings"] # "created_at"
                
            ).flatten(limit=1500)
            
            for thread in replies:
                thread_user = np.int64(thread.author_id)
                thread_setting = thread.reply_settings
                thread_tweet_id = np.int64(thread.id)
                referenced_tweet_id = np.nan
                for referenced_tweet in thread.referenced_tweets:
                    if referenced_tweet.type == 'replied_to':
                        referenced_tweet_id = np.int64(referenced_tweet.id)
                        break
                
                threads["conv_id"].append(conv_id)
                threads["uid"].append(thread_user)
                threads["tweet_id"].append(thread_tweet_id)
                threads["reply_setting"].append(thread_setting)
                threads["parent_tweet_id"].append(referenced_tweet_id)
                    
        except KeyboardInterrupt:
            print("current index: ", idx)
            print("current convid: ", conv_id)
            pdb.set_trace()
        except Exception as e:
            eprint(e)
        
        save_data(threads, f'threads_fast_{which}')
            
    save_data(threads, f'threads_fast_{which}')