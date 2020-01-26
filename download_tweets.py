import json
from functools import reduce

import ndjson
import tweepy
import os
import tqdm
from tweepy import TweepError

CONSUMER_KEY = os.environ.get("CONSUMER_KEY", None)
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET", None)
ACCESS_KEY = os.environ.get("ACCESS_KEY", None)
ACCESS_KEY_SECRET = os.environ.get("ACCESS_KEY_SECRET", None)

def download_tweet_list(api, output, ids):
    res = []
    for i in tqdm.tqdm(range(len(ids)//100 + 1)):
        statuses = api.statuses_lookup(ids[i*100:(i+1)*100],  tweet_mode="extended")
        res.extend([status._json for status in statuses])
    with open(output,"w") as output:
        ndjson.dump(res, output)


def download_user_history(api, output_name, screen_name=None, user_id=None, since_id=None, exclude_replies=False,
                          save_retweeters=False):
    res = []
    for page in tweepy.Cursor(api.user_timeline, screen_name=screen_name, user_id=user_id, tweet_mode="extended",
                              since_id=since_id, count=200, exclude_replies=exclude_replies).pages():
        res.extend(page)
    res = [item._json for item in res]

    with open(output_name, "w") as output:
        ndjson.dump(res, output)

    if save_retweeters:
        print("Extracting retweeters")
        users = {}
        for item in res:
            users[item["id"]] = api.retweeters(item["id"])
        with open(output_name + ".retweets.json", "w") as output:
            json.dump(users, output)
        return users


def download_users_and_retweeters(api, output_dir, screen_name="realDonaldTrump", resume=True):
    retweeters_folder = os.path.join(output_dir, "retweeters")
    followers_folder = os.path.join(output_dir, "followers")
    friends_folder = os.path.join(output_dir, "friends")
    for folder in [output_dir, retweeters_folder, followers_folder, friends_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print("Downloading Tweets")
    retweeter_list = os.path.join(output_dir, screen_name + ".ndjson.retweets.json")
    if not resume or not os.path.exists(retweeter_list):
        users = download_user_history(api, os.path.join(output_dir, screen_name + ".ndjson"), screen_name=screen_name,
                                      save_retweeters=True)
    else:
        users = json.load(open(retweeter_list, "r"))

    users = reduce(set.union, users.values(), set())
    users = users - set(int(fname.split(".")[0]) for fname in os.listdir(retweeters_folder))
    print("Downloading Retweeters")
    for user in tqdm.tqdm(users):
        try:
            download_user_history(api, os.path.join(retweeters_folder, str(user) + ".ndjson"), user_id=user,
                                  save_retweeters=False)
        except TweepError as e:
            print("Failed for user: {}".format(user))
            print(e)

    print("Downloading Followers")
    for follower in tweepy.Cursor(api.followers, screen_name=screen_name, count=200).items():
        download_user_history(api, os.path.join(followers_folder, str(follower.id) + ".ndjson"), user_id=follower.id,
                              save_retweeters=False)

    print("Downloading Friends")
    for friend in tweepy.Cursor(api.friends, screen_name=screen_name, count=200).items():
        download_user_history(api, os.path.join(friends_folder, str(friend.id) + ".ndjson"), user_id=friend.id,
                              save_retweeters=False)


if __name__ == "__main__":
    import pandas as pd

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_KEY_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    #df = pd.read_csv("https://raw.githubusercontent.com/ENCASEH2020/hatespeech-twitter/master/hatespeech_labels.csv")
    #download_tweet_list(api, "found_tweets.txt", list(df["tweet_id"].values))

    download_users_and_retweeters(api, r"C:\Develop\causal_inference_project\data\crawling", screen_name="realDonaldTrump")