import json
from functools import reduce

import ndjson
import tweepy
import os
import tqdm

CONSUMER_KEY = os.environ.get("CONSUMER_KEY", None)
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET", None)
ACCESS_KEY = os.environ.get("ACCESS_KEY", None)
ACCESS_KEY_SECRET = os.environ.get("ACCESS_KEY_SECRET", None)

def download_tweet_list(api, output, ids):
    res = []
    for i in tqdm.tqdm(range(len(ids)//100 + 1)):
        statuses = api.statuses_lookup(ids[i:i+100],  tweet_mode="extended")
        res.extend([status._json for status in statuses])
    with open(output,"w") as output:
        ndjson.dump(res, output)


def download_user_history(api, output_name, screen_name=None, user_id=None, since_id=None, exclude_replies=False,
                          save_retweeters=False):
    res = []
    users = {}
    for page in tweepy.Cursor(api.user_timeline, screen_name=screen_name, user_id=user_id, tweet_mode="extended",
                              since_id=since_id, count=200, exclude_replies=exclude_replies).pages():
        res.extend([item._json for  item in page])
        if save_retweeters:
            for item in page:
                users[item.id] = api.retweeters(item.id)

    with open(output_name, "w") as output:
        ndjson.dump(res, output)
    if save_retweeters:
        with open(output_name + ".retweets.json", "w") as output:
            json.dump(users, output)
    return users


def download_users_and_retweeters(api, output_dir, screen_name="realDonaldTrump"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "retweeters")):
        os.makedirs(os.path.join(output_dir, "retweeters"))
    if not os.path.exists(os.path.join(output_dir, "followers")):
        os.makedirs(os.path.join(output_dir, "followers"))
    if not os.path.exists(os.path.join(output_dir, "friends")):
        os.makedirs(os.path.join(output_dir, "friends"))

    print("Downloading Tweets")
    users = download_user_history(api, os.path.join(output_dir, screen_name + ".ndjson"), screen_name=screen_name,
                                  save_retweeters=True)

    print("Downloading Retweeters")
    users = reduce(set.union, users.values(), set())
    for user in tqdm.tqdm(users):
        download_user_history(api, os.path.join(output_dir, "retweeters", str(user) + ".ndjson"), user_id=user,
                              save_retweeters=False)

    print("Downloading Followers")
    for follower in tweepy.Cursor(api.followers, screen_name=screen_name, count=200).items():
        download_user_history(api, os.path.join(output_dir, "followers", str(follower.id) + ".ndjson"), user_id=follower.id,
                              save_retweeters=False)

    print("Downloading Friends")
    for friend in tweepy.Cursor(api.friends, screen_name=screen_name, count=200).items():
        download_user_history(api, os.path.join(output_dir, "friends", str(friend.id) + ".ndjson"), user_id=friend.id,
                              save_retweeters=False)


if __name__ == "__main__":
    import pandas as pd

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_KEY_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    #df = pd.read_csv("https://raw.githubusercontent.com/ENCASEH2020/hatespeech-twitter/master/hatespeech_labels.csv")
    #download_tweet_list(api, "found_tweets.txt", list(df["tweet_id"].values))

    download_users_and_retweeters(api, r"C:\Develop\causal_inference_project\data\crawling", screen_name="realDonaldTrump")