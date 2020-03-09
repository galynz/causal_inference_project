import datetime

import bisect
import dateutil
import shutil

import dask.bag as db
from transformers import pipeline
import json
import os
import glob
from dask.distributed import Client, progress, get_client, LocalCluster
import re
import pandas as pd

FINAL_LINK = re.compile(r"https://t\.co/\w+$")

class HierarchicalPath(os.PathLike):
    """
    Utility class for easy access of files and folders
    """
    def __init__(self, base_path):
        self.base_path = base_path

    def __getitem__(self, item):
        return HierarchicalPath(os.path.join(self.base_path, item))

    def __str__(self):
        return self.base_path

    def __fspath__(self):
        return self.base_path


class TweeterBag(db.Bag):
    def repartition(self, npartitions):
        return self.__class__(*super(TweeterBag, self).repartition(npartitions)._args)

    def checkpoint(self, output_path, max_workers=None, overwrite=False):
        if overwrite:
            shutil.rmtree(output_path, ignore_errors=True)
        if not os.path.exists(output_path):
            print(f"Saving in {output_path}")
            client = get_client()
            if max_workers is not None and hasattr(client, "cluster"):
                num_workers = len(client.cluster.worker_spec)
                client.cluster.scale(max_workers)

            self.map(json.dumps).to_textfiles(f"{output_path}/*.ndjson")#, name_function=self._name_function)

            if max_workers is not None and hasattr(client, "cluster"):
                client.cluster.scale(num_workers)
        print(f"Reloading from {output_path}")
        return read_tweets(output_path)

    def sentiment_analysis(self):
        analyzer_bag = db.from_sequence([
            pipeline("sentiment-analysis", device=0)
        ])

        def analyze_tweet(pair):
            tweet, analyzer = pair
            tweet["display_text"] = extract_text(tweet)
            sentiment = analyzer(tweet["display_text"])[0]
            sentiment["score"] = float(sentiment["score"])
            return {"tweet": tweet, "sentiment": sentiment}

        return self.product(analyzer_bag).map(analyze_tweet)

    def extract_features(self, prefix):
        def extract_features(tweet):
            base = {}
            if "tweet" in tweet:
                base.update({f"{prefix}_{k}": v for k, v in tweet["sentiment"].items()})
                tweet = tweet["tweet"]

            base.update({
                f"{prefix}_status_id": tweet["id_str"],
                f"{prefix}_user_id": tweet["user"]["id_str"],
                f"{prefix}_screen_name": tweet["user"]["screen_name"],
                f"{prefix}_display_text": tweet["display_text"],
                f"{prefix}_created_at": tweet["created_at"],
            })

            if tweet["in_reply_to_status_id_str"] is not None:
                base[f"interaction_type"] = "reply"
                base[f"interaction_status_id"] = tweet["in_reply_to_status_id_str"]
                base[f"interaction_screen_name"] = tweet["in_reply_to_screen_name"]

            elif "quoted_status" in tweet:
                base["interaction_type"] = "quote"
                base[f"interaction_status_id"] = tweet.get("quoted_status", {}).get("id_str")
                base[f"interaction_screen_name"] = tweet.get("quoted_status", {}).get("user", {}).get("screen_name")
                base[f"interaction_display_text"] = extract_text(tweet.get("quoted_status", {}))
                base[f"interaction_created_at"] = tweet.get("quoted_status", {}).get("created_at")

            elif "retweeted_status" in tweet:
                base["interaction_type"] = "retweet"
                base[f"interaction_status_id"] = tweet.get("retweeted_status", {}).get("id_str")
                base[f"interaction_screen_name"] = tweet.get("retweeted_status", {}).get("user", {}).get("screen_name")
                base[f"interaction_display_text"] = extract_text(tweet.get("retweeted_status", {}))
                base[f"interaction_created_at"] = tweet.get("retweeted_status", {}).get("created_at")

            for field in ["screen_name", "followers_count", "friends_count", "listed_count",
                          "created_at", "statuses_count", "location"]:
                base[f"{prefix}_user_{field}"] = tweet["user"][field]
            return base
        return self.map(extract_features)


def read_tweets(path):
    names = glob.glob(f"{path}/*.ndjson")
    return TweeterBag(*db.read_text(names)._args).map(json.loads)


def extract_text(tweet):
    if tweet == {}:
        return None
    start, end = tweet["display_text_range"]
    return re.sub(FINAL_LINK, "", tweet["full_text"][start:end]).strip()


def prepare_data_reply_analysis(input_path, output_path, overwrite_all=False):
    trump_tweets = (
        read_tweets(input_path["trump"])
            .sentiment_analysis()
            .extract_features("trump")
            .checkpoint(output_path["trump_sentiment"], 2, overwrite=overwrite_all)
            .map(lambda row: {k: v for k, v in row.items() if k.startswith("trump")})
    )

    retweeters = (
        read_tweets(input_path["retweeters"])
            .filter(
            lambda tweet: (tweet["in_reply_to_status_id"] is not None and
                           tweet["in_reply_to_screen_name"] == "realDonaldTrump") or
                          (tweet["is_quote_status"] == True and
                           tweet.get("retweeted_status", {}).get("user", {})
                           .get("screen_name") == "realDonaldTrump"),
            )
            .sentiment_analysis()
            .extract_features("tweet")
            .repartition(1)
            .checkpoint(output_path["only_retweets_sentiments"], 2, overwrite=overwrite_all)
    )

    data = (
        retweeters
            .filter(lambda tweet: tweet.get("interaction_type") in ("reply", "quote"))
            .join(
                trump_tweets.map(lambda tweet: {k: v for k, v in tweet.items() if k.startswith("trump")}),
                "interaction_status_id", "trump_status_id"
            )
            .map(lambda pair: dict(**pair[0], **pair[1]))
            .checkpoint(output_path["only_retweets_sentiments_with_trump"], overwrite=overwrite_all)
    )
    return pd.DataFrame(data.compute())

def prepare_data_before_and_after(input_path, output_path, time_buffer=5*3600, overwrite_all=False):
    trump_tweets = (
        read_tweets(input_path["trump"])
            .sentiment_analysis()
            .extract_features("trump")
            .checkpoint(output_path["trump_sentiment"], 2, overwrite=overwrite_all)
            .map(lambda row: {k: v for k, v in row.items() if k.startswith("trump")})
    )

    ordered_tweets = read_tweets(input_path["trump"]).map(
        lambda row: (dateutil.parser.parse(row["created_at"]), row["id"])
    ).compute()
    ordered_tweets = sorted(ordered_tweets, key=lambda x: x[0])

    def get_tweets(row):
        row_time = dateutil.parser.parse(row["created_at"])
        diff = datetime.timedelta(seconds=time_buffer)
        start = bisect.bisect_left(ordered_tweets, (row_time - diff,))
        end = bisect.bisect_right(ordered_tweets, (row_time + diff,))
        return {"tweet": row, "trump_tweets_id": [x[1] for x in ordered_tweets[start:end]]}

    retweeters = (
        read_tweets(input_path["retweeters"])
            .map(get_tweets)
            .filter(lambda row: len(row["trump_tweets_id"]) > 0)
            .map(lambda row: row["tweet"])
            .sentiment_analysis()
            .extract_features("tweet")
            .checkpoint(output_path["before_and_after_sentiments"], 2, overwrite=True)
    )

    data = (
        retweeters
            .filter(lambda tweet: tweet.get("interaction_type") in ("reply", "quote"))
            .join(
                trump_tweets.map(lambda tweet: {k: v for k,v in tweet.items() if k.startswith("trump")}),
                "interaction_status_id", "trump_status_id"
            )
            .map(lambda pair: dict(**pair[0], **pair[1]))
            .checkpoint(output_path["before_and_after_sentiments_with_trump"], overwrite=True)
    )
    return data

if __name__ == "__main__":
    client = Client(threads_per_worker=2, memory_limit='6GB')
    input_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/input")
    output_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/output_pipeline")
    prepare_data_reply_analysis(input_path, output_path, overwrite_all=False)
    prepare_data_before_and_after(input_path, output_path, time_buffer=5 * 3600, overwrite_all=False)
    client.close()
