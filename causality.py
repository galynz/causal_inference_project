"""
Contain all the analysis code for the project
"""

import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.special

from prepare_data import prepare_data_reply_analysis, prepare_data_before_and_after


def plot_random_effects_hist_anad_qqplot(fitted_model):
    sds = pd.Series(np.exp(fitted_model.vcp_mean), index=fitted_model.model.vcp_names)
    plt.figure(figsize=(7, 7))
    for i, effect in enumerate(fitted_model.model.vcp_names):
        e = fitted_model.random_effects(effect)

        ax = plt.subplot(len(fitted_model.model.vcp_names), 2, 2 * i + 1)
        _ = qqplot(e["Mean"] / sds[effect], line='45', ax=ax)
        plt.title(f"{effect} Posterior QQPlot")
        plt.grid()
        sns.despine()
        plt.subplot(len(fitted_model.model.vcp_names), 2, 2 * i + 2)
        e["SD"].hist()
        plt.title(f"{effect} Posterior Standard Deviation")
        sns.despine()
    plt.tight_layout()
    plt.show()


def backdoor_cate(Y_cond, X, G, T, U=None):
    # Dividing by |T(g)| = |{T(g,x=1)} u {T(g,x=0)}|
    group_by = [G, X]
    if U is not None:
        group_by.insert(0, U)

    Y_cond = Y_cond.to_frame("mean")
    Y_cond["var"] = Y_cond["mean"]*(1-Y_cond["mean"])
    backdoor_estimate = Y_cond.reset_index(T).groupby(group_by).agg(
        {T: "nunique", "mean": "sum", "var": "sum"}
    )
    mean_estimate = (backdoor_estimate["mean"]/backdoor_estimate[T]).unstack().fillna(0).stack()
    var_estimate = (backdoor_estimate["var"]/np.square(backdoor_estimate[T])).unstack().fillna(0).stack()

    # Dividing by |G|
    group_by = [X]
    if U is not None:
        group_by.insert(0, U)
    mean_estimate = mean_estimate.groupby(group_by).mean()
    var_estimate = var_estimate.groupby(group_by).agg(["mean", "count"])
    var_estimate = (var_estimate["mean"] / var_estimate["count"])
    if U is not None:
        mean_estimate = mean_estimate.unstack()
        var_estimate = var_estimate.unstack()

    ate = mean_estimate[1] - mean_estimate[0]
    return ate, 2*np.sqrt(var_estimate[0] + var_estimate[1])


def fit_mixed_model(formula, random, data, debug=False):
    model = sm.BinomialBayesMixedGLM.from_formula(formula, random, data, vcp_p=10)

    return model.fit_vb(sd=np.hstack([
        np.ones(model.k_fep),
        2 * np.ones(model.k_vcp),
        np.ones(model.k_vc)
    ]), fit_method="Newton-CG", minim_opts={"maxiter": 10000000, "disp": debug}, verbose=debug)


def estimate_ate(data, X, Y, G, T, X_time_field, Y_time_field,
                                Y_minimum_time_difference):
    """
    This function calculates the ATE based on the backdoor criteria (Pearl).
    Assuming the following causal graph:
          Time Group ---> Treatment ----------------
               |                 |                  |
               |                 |                  |
               v                 v                  v
                ---------> Population Mood  ---> User Response
    :param data:
    :param treatment_field:
    :param response_field:
    :param treatment_time_field:
    :return:
    """
    data = data.copy()

    # We consider tweets that are too close to Trump tweet are unstable,
    # as the "population mode" was stabilized (they determine the mood)
    print(
        "Dropping too close to treatment",
        (data[Y_time_field] - data[X_time_field] < Y_minimum_time_difference).agg(["mean", "sum"])
    )
    data = data[data[Y_time_field] - data[X_time_field] > Y_minimum_time_difference].copy()

    # Calculate conditional P(Y = 1| G, T, X)
    data_grouped = data.groupby([G, T, X])[Y].agg("mean")

    # Calculate ATE
    return backdoor_cate(data_grouped, X, G, T)

def estimate_ate_by_glm(df):
    model = sm.GLM.from_formula(
        "tweet_label ~ 1 + trump_label + C(time_group)",
        df,
        family=sm.families.Binomial(sm.families.links.probit()),
    )
    fitted = model.fit()

    estimates = pd.Series(fitted.fittedvalues, index=df.index) \
        .groupby([df["time_group"], df["trump_label"]]) \
        .agg("mean").unstack().fillna(0)
    estimates = estimates[estimates.notnull().all(axis=1)]
    return (estimates[1] - estimates[0]).mean()


def estimate_ate_before_and_after(before_after):
    diff = before_after[1] - before_after[0]
    diff_var = before_after[1] * (1 - before_after[1]) + before_after[0] * (1 - before_after[0])

    label_effect = diff.groupby(["trump_label", "trump_status_id"]).mean().groupby("trump_label").mean()
    mean = label_effect[1] - label_effect[0]

    diff_var_group = diff_var.groupby(["trump_label", "trump_status_id"])
    user_var = diff_var_group.sum() / np.square(diff_var_group.count())

    user_var_group = user_var.groupby(["trump_label"])
    label_var = user_var_group.sum() / np.square(user_var_group.count())

    return mean, label_var.sum()


if __name__ == "__main__":
    from dask.distributed import Client
    from prepare_data import HierarchicalPath

    print("##########################")
    print("Reply analysis:")
    print("==========================")

    client = Client()
    input_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/input")
    output_path = HierarchicalPath("/media/tmrlvi/My Passport/Research/causal_inference_data2/output_pipeline")
    df = prepare_data_reply_analysis(input_path, output_path, overwrite_all=False)

    client.close()

    df["tweet_created_at"] = pd.to_datetime(df["tweet_created_at"])
    df["trump_created_at"] = pd.to_datetime(df["trump_created_at"])
    df["time_group"] = df["trump_created_at"].dt.round("1D").astype("str")
    # Make treatment 0/1 (control/treatment)
    df["trump_label"] = (df["trump_label"] == "NEGATIVE").astype("int")
    df["tweet_label"] = (df["tweet_label"] == "NEGATIVE").astype("int")

    # Estimating ATE using the backdoor criterion
    print(
        "Backdoor criterion ATE:",
          estimate_ate(
              df, "trump_label", "tweet_label", "time_group", "trump_status_id", "trump_created_at", "tweet_created_at",
              datetime.timedelta(minutes=10)
          )
    )

    # Removing users without enough exposure
    df2 = df.merge(
        df.groupby("tweet_user_id")[["trump_label", "trump_status_id"]].nunique().rename(
            {"trump_label": "unique_labels", "trump_status_id": "unique_exposures"}, axis=1
        ).query("(unique_labels>1) & (unique_exposures > 1)"),
        on="tweet_user_id",
        how="inner"

    ).drop(["unique_labels", "unique_exposures"], axis=1)

    print(
        "GLM estimate of ATE (1 day time group):",
        estimate_ate_by_glm(df2)
    )
    df2["time_group"] = df2["trump_created_at"].dt.round("1H").astype("str")
    print(
        "GLM estimate of ATE (1 hour time group):",
        estimate_ate_by_glm(df2)
    )

    res = fit_mixed_model(
        "tweet_label ~ 1 + trump_label", # + C(time_group) + C(tweet_user_id) + C(tweet_user_id):trump_label",
        {
            "Group": "0 + C(time_group)",
            "User Bias": "0 + C(tweet_user_id)",
            "User Effect": "0 + C(tweet_user_id): trump_label"
        },
        df2
    )

    print(
        "Average ITE (using random effects):",
        scipy.special.expit(res.fe_mean.sum()) - scipy.special.expit(res.fe_mean[0])
    )
    plot_random_effects_hist_anad_qqplot(res)

    print("##########################")
    print("Before and After Analysis")
    print("==========================")

    df = prepare_data_before_and_after(input_path, output_path, overwrite_all=False)

    print(len(df))
    df["trump_created_at"] = pd.to_datetime(df["trump_created_at"])
    df["tweet_created_at"] = pd.to_datetime(df["tweet_created_at"])
    df = df.assign(
        trump_label=lambda df: (df["trump_label"] == "NEGATIVE").astype("int"),
        tweet_label=lambda df: (df["tweet_label"] == "NEGATIVE").astype("int"),
        is_after=lambda df: (df["tweet_created_at"] > df["trump_created_at"]).astype("int")
    )

    before_and_after = df.join(
        df.groupby(["tweet_user_id", "trump_status_id"])["is_after"].nunique().to_frame('user_count').query(
            'user_count > 1'),
        on=["tweet_user_id", "trump_status_id"], how="inner"
    )
    print(df["tweet_user_id"].nunique())
    print(before_and_after["trump_status_id"].nunique())

    before_after = before_and_after.groupby(["trump_label", "trump_status_id", "tweet_user_id", "is_after"])[
        "tweet_label"].mean().unstack("is_after")

    ate, std = estimate_ate_before_and_after(before_after)
    print(f"ATE: {ate} (+- {2 * std})")

    diff = before_after[1] - before_after[0]
    diff_var = before_after[1] * (1 - before_after[1]) + before_after[0] * (1 - before_after[0])

    label_effect = diff.groupby(["trump_label", "trump_status_id"]).mean().groupby("trump_label").mean()
    mean = label_effect[1] - label_effect[0]

    diff_var_group = diff_var.groupby(["trump_label", "trump_status_id"])
    user_var = diff_var_group.sum() / np.square(diff_var_group.count())

    user_var_group = user_var.groupby(["trump_label"])
    label_var = user_var_group.sum() / np.square(user_var_group.count())

    print(f"ATE: {mean} (+- {2*label_var.sum()})")

    res = fit_mixed_model(
        "tweet_label ~ 1 + is_after + trump_label:is_after",
        {
            "Trump Tweet": "0 + C(trump_status_id)",
            "User baseline": "0 + C(tweet_user_id)",
            "User Bias": "0 + C(tweet_user_id):is_after",
            "User Effect": "0 + C(tweet_user_id):trump_label:is_after"
        },
        before_and_after
    )

    print("Average ITE: ",
            scipy.special.expit(res.fe_mean[0:1].sum()) -
            scipy.special.expit(res.fe_mean[0].sum()) -
            scipy.special.expit(res.fe_mean.sum()) +
            scipy.special.expit(res.fe_mean[0].sum())
    )
    print(res.summary())
    plot_random_effects_hist_anad_qqplot(res)