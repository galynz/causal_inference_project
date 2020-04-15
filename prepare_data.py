"""
Prepares the data for analysis. Heavy script that use parallelization to speed it up.
"""
import datetime
from dateutil import parser

import bisect
import dateutil
import shutil

import dask.bag as db
from functools import partial
from transformers import pipeline
import json
import os
import glob
from dask.distributed import Client, progress, get_client, LocalCluster
import re
import pandas as pd
import sys

FINAL_LINK = re.compile(r"https://t\.co/\w+$")


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        else:
            return json.JSONEncoder.default(self, obj)

class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, source):
        for key, value in source.items():
            try:
                source[key] = parser.parse(value)
            except:
                pass
        return source


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

            try:
                self.map(lambda x: json.dumps(x, cls=Encoder)).to_textfiles(f"{output_path}/*.ndjson")
            except ValueError():
                shutil.rmtree(output_path, ignore_errors=True)
                print(f"Failed saving. Removing path ({output_path}).")
                raise

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


def is_response_to_trump(tweet):
    if tweet["in_reply_to_status_id_str"] is not None:
        return tweet["in_reply_to_screen_name"] == "readDonaldTrump"

    elif "quoted_status" in tweet:
        return tweet.get("quoted_status", {}).get("user", {}).get("screen_name") == "realDonaldTrump"

    elif "retweeted_status" in tweet:
        return tweet.get("retweeted_status", {}).get("user", {}).get("screen_name") == "realDonaldTrump"

    return False


def get_response_id(tweet):
    if tweet["in_reply_to_status_id_str"] is not None:
        return tweet["in_reply_to_status_id_str"]

    elif "quoted_status" in tweet:
        return tweet.get("quoted_status", {}).get("id_str")

    elif "retweeted_status" in tweet:
        return tweet.get("retweeted_status", {}).get("id_str")

def dataframe_to_tweeter_bag(df):
    return TweeterBag(*df.to_bag()._args).map(lambda x: dict(zip(df.columns, x)))


def extract_text(tweet):
    if tweet == {}:
        return None
    start, end = tweet["display_text_range"]
    return re.sub(FINAL_LINK, "", tweet["full_text"][start:end]).strip()


def add_trump_status_id(pair):
    tweet, trump_before, trump_after = pair
    tweet["tweet"]["trump_before_status_id"] = trump_before[1]
    tweet["tweet"]["trump_before_status_time"] = trump_before[0]
    tweet["tweet"]["trump_after_status_id"] = trump_after[1]
    tweet["tweet"]["trump_after_status_time"] = trump_after[0]
    return tweet["tweet"]


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


def prepare_data_before_and_after(input_path, output_path, time_buffer=5 * 3600, overwrite_all=False):
    trump_tweets = (
        read_tweets(input_path["trump"])
            .sentiment_analysis()
            .extract_features("trump")
            .checkpoint(output_path["trump_sentiment"], 2, overwrite=overwrite_all)
            .map(lambda row: {k: v for k, v in row.items() if k.startswith("trump")})
    )

    # Get all trump interactions for a user
    interaction_by_user = (
        read_tweets(input_path["retweeters"])
            .filter(is_response_to_trump)
            .map(
            lambda x: (x["user"]["id_str"], {(x["created_at"], get_response_id(x))})
        ).foldby(lambda x: x[0], lambda x, y: (x[0], x[1] | y[1]))
            .map(lambda x: (x[1][0], list(x[1][1])))
            .checkpoint(output_path["interaction_by_user"], overwrite=overwrite_all)
            .map(
            lambda x: {"tweet_user_id": x[0], "trump_interactions": [[parser.parse(y[0]), y[1]] for y in x[1]]}
        )
    )

    retweeters = (
        dataframe_to_tweeter_bag(
            read_tweets(input_path["retweeters"]).map(lambda x: {
                "timestamp": dateutil.parser.parse(x["created_at"]),
                "tweet": x,
                "tweet_user_id": x["user"]["id_str"]
            }).to_dataframe().merge(
                interaction_by_user.to_dataframe(), on="tweet_user_id",
            )
        ).checkpoint(output_path["tweets_trump_product"], max_workers=2, overwrite=overwrite_all).map(
            lambda x: (
                x,
                min(
                    filter(lambda p: dateutil.parser.parse(p[0]) <= dateutil.parser.parse(x["timestamp"]), x["trump_interactions"]),
                    default=["1970-01-01T00:00:00Z00:00", None],
                    key=lambda p: abs((dateutil.parser.parse(p[0]) - dateutil.parser.parse(x["timestamp"])).total_seconds())
                ),
                min(
                    filter(lambda p: dateutil.parser.parse(p[0]) > dateutil.parser.parse(x["timestamp"]), x["trump_interactions"]),
                    default=["1970-01-01T00:00:00Z00:00", None],
                    key=lambda p: abs((dateutil.parser.parse(p[0]) - dateutil.parser.parse(x["timestamp"])).total_seconds())
                )
            )
        ).filter(
            lambda x: abs((dateutil.parser.parse(x[1][0]) - dateutil.parser.parse(x[0]["timestamp"])).total_seconds()) < 1 * 3600 or abs(
                (dateutil.parser.parse(x[2][0]) - dateutil.parser.parse(x[0]["timestamp"])).total_seconds()) < 1 * 3600
        ).map(add_trump_status_id)
            .checkpoint(output_path["before_and_after_filtered_single"], max_workers=2, overwrite=overwrite_all)
            .sentiment_analysis()
            .checkpoint(output_path["before_and_after_filtered_single_with_sentiment"], overwrite=overwrite_all)
    )

    data = (
        retweeters.filter(lambda tweet: tweet["in_reply_to_status_id_str"] is not None or "quoted_status" in tweet)
            .join(trump_tweets, "trump_status_id", "id_str")
            .filter(
            lambda pair: abs(
                (dateutil.parser.parse(pair[0]["created_at"]) -
                 dateutil.parser.parse(pair[1]["created_at"])).total_seconds()
            ) < 10 * 60)
            .map(lambda pair: pair[1])
            .sentiment_analysis()
            .extract_features()
            .checkpoint(output_path["before_and_after_filtered_10mins"], overwrite=overwrite_all)
            .join(trump_tweets, "trump_status_id", "trump_status_id")
            .checkpoint(output_path["before_and_after_filtered_10mins_w_sentiment/"], overwrite=overwrite_all)
    )
    return pd.DataFrame(data.compute())


if __name__ == "__main__":
    client = Client(threads_per_worker=2, memory_limit='6GB')
    input_path = HierarchicalPath(sys.argv[1])
    output_path = HierarchicalPath(sys.argv[1])
    prepare_data_reply_analysis(input_path, output_path, overwrite_all=False)
    prepare_data_before_and_after(input_path, output_path, time_buffer=5 * 3600, overwrite_all=False)
    client.close()
