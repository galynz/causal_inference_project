import datetime
import numpy as np
import pandas as pd

from prepare_data import prepare_data_reply_analysis


def backdoor_binary_respose_ate(data, treatment_field, treatment_label,
                                response_field, response_label,
                                treatment_time_field, response_time_field,
                                response_minimum_time_difference,
                                time_resolution):
    """
    This function calculates the ATE based on the backdoor criteria (Pearl).
    Assuming the following causal graph:
          Time Group ---> Treatment ----------------
               |                 |                  |
               |                 |                  |
               v                 v                  v
                ---------> Population Mood  ---> User Respose
    :param data:
    :param treatment_field:
    :param response_field:
    :param treatment_time_field:
    :return:
    """
    data = data.copy()

    data[treatment_time_field] = pd.to_datetime(data[treatment_time_field])
    data[response_time_field] = pd.to_datetime(data[response_time_field])

    # We consider tweets that are too close to Trump tweet are unstable,
    # as the "population mode" was stabilized (they determine the mood)
    # TODO: find a better way to report
    print(
        "Dropping too close to treatment",
        (data[response_time_field] - data[treatment_time_field] < response_minimum_time_difference).agg(["mean", "sum"])
    )
    data = data[data[response_time_field] - data[treatment_time_field] > response_minimum_time_difference].copy()

    data[treatment_time_field] = data[treatment_time_field].dt.round(time_resolution)
    # Make treatment 0/1 (control/treatment)
    data[treatment_field] = (data[treatment_field] == treatment_label).astype("int")
    data[response_field] = (data[response_field] == response_label).astype("int")

    # Calculate conditional P(UR | TG, Tr)
    data_grouped = data.groupby([treatment_time_field, treatment_field, response_field])

    data_counts = data_grouped[treatment_time_field].count().unstack()
    data_counts = data_counts.sum(axis=1).to_frame("count").join((data_counts.T / data_counts.sum(axis=1)).T)

    # Calculate ATE
    print(
        "Dropping due to lack of retweets",
        (data_counts.isnull().any(axis=1)).agg(["mean", "sum"])
    )
    # Assuming uniform prior on time groups
    backdoor_estimate = data_counts[
        ~(data_counts.isnull().any(axis=1)) #TODO: report how many were dropped
    ].groupby(treatment_field).agg(
        {"count": "sum", 1: "mean"}
    )

    std = np.sqrt((backdoor_estimate[1] * (1-backdoor_estimate[1])) / backdoor_estimate["count"]).sum()
    ate = backdoor_estimate.loc[1,1] - backdoor_estimate.loc[0,1]
    return ate, std

if __name__ == "__main__":
    from dask.distributed import Client
    from prepare_data import HierarchicalPath

    client = Client()
    input_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/input")
    output_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/output_pipeline")
    data_df_full_filtered_bag = prepare_data_reply_analysis(input_path, output_path, overwrite_all=True)

    data_df_full_filtered = pd.DataFrame(data_df_full_filtered_bag.compute())
    client.close()

    backdoor_binary_respose_ate(data_df_full_filtered, "sentiment_label_trump", "NEGATIVE",
                                "sentiment_label_retweet", "NEGATIVE", "created_at_trump", "created_at_retweet",
                                datetime.timedelta(minutes=30), "1D")