import ndjson
import tweepy
import os
import tqdm

CONSUMER_KEY = os.environ.get("CONSUMER_KEY", None)
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET", None)
ACCESS_KEY = os.environ.get("ACCESS_KEY", None)
ACCESS_KEY_SECRET = os.environ.get("ACCESS_KEY_SECRET", None)

def download_tweet_list(api, ids, output):
    res = []
    for i in tqdm.tqdm(range(len(ids)//100 + 1)):
        statuses = api.statuses_lookup(ids[i:i+100],  tweet_mode="extended")
        res.extend([status._json for status in statuses])
    with open(output,"wb") as output:
        ndjson.dump(res, output)

def download_user_history(api, user):
    # def limit_handled(cursor):
    #     while True:
    #         try:
    #             yield cursor.next()
    #         except tweepy.RateLimitError:
    #             time.sleep(15 * 60)
    #API.user_timeline([id/user_id/screen_name][, since_id][, max_id][, count][, page])
    #tweepy.Cursor(api.user_timeline, id="twitter").pages():
    pass

def map_user_frequent_retweeters(api, user):
    pass

if __name__ == "__main__":
    import pandas as pd

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_KEY_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    df = pd.read_csv("https://raw.githubusercontent.com/ENCASEH2020/hatespeech-twitter/master/hatespeech_labels.csv")
    download_tweet_list(api, list(df["tweet_id"].values), "found_tweets.txt")