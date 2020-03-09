import openpyxl
from datetime import datetime
from dask.distributed import Client
import scipy.stats
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import statsmodels.api as sm

from causality import backdoor_binary_respose_ate
from prepare_data import HierarchicalPath, prepare_data_reply_analysis, prepare_data_before_and_after

def reply_analysis_report(data_input_path, data_output_path):
    reply_analysis = prepare_data_reply_analysis(data_input_path, data_output_path)

    score = reply_analysis.assign(
        tweet_negative_score = lambda df: df.apply(
            lambda x: x["tweet_score"] if x["tweet_label"] == "NEGATIVE" else 1 - x["tweet_score"], axis=1
        ),
        trump_negative_score = lambda df: df.apply(
        lambda x: x["trump_score"] if x["trump_label"] == "NEGATIVE" else 1 - x["trump_score"], axis=1
        )
    )
    logits = scipy.special.logit(score[["negative_score_retweet", "negative_score_trump"]])
    logits.plot(kind="scatter", x="negative_score_trump", y="negative_score_retweet", alpha=0.1)
    logits.save("plots/logit_sentiment_score.png")
    print("Naive Sentiment Score Calculation",
          scipy.stats.pearsonr(logits["negative_score_trump"], logits["negative_score_retweet"])
    )

    tmp_data = reply_analysis[~reply_analysis["trump_label"].isnull()]
    x = sm.add_constant(tmp_data[[
        # "created_at_trump_day", "created_at_trump_month", "created_at_trump_year",
        "followers_count_norm", "friends_count_norm", "listed_count_norm", "statuses_count_norm"]])
    res = GLM(tmp_data['trump_label'].astype("category").cat.codes, x,
              family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
    print(res.summary())

    print("ATE", backdoor_binary_respose_ate(
        reply_analysis, "trump_label", "NEGATIVE", "tweet_label", "NEGATIVE", "trump_created_at", "tweet_created_at",
        datetime.timedelta(minutes=0), "1D"
    ))

if __name__ == "__main__":
    client = Client()

    input_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/input")
    output_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/output_pipeline")

    reply_analysis_report

    before_and_after = prepare_data_before_and_after(input_path, output_path)

    client.close()