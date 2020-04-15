# Casual Inference Project - Trump Tweets Effect on Retweeters

This repository contains code and bits related to a causality project aimed at 
analysing Trump's twitter-shpere. It contains raw analysis notebooks and scripts, 
alongside scripts that allow reproduction of the analysis.

To reproduce our results:

  1. Use `download_tweets.py` to curate a dataset of trump tweets and retweeters.        
    - You need CONSUMER_KEY and ACCESS_KEY for you "app" on tweiiter.
  1. Run `prepare_data.py` with the relevant input and output paths. You might 
  need a strong machine for running it.
  1. Run `causality.py` to run our analysis.