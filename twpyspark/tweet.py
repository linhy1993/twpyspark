import json
import re

import tweepy
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from twpyspark.settings import Config


def _clean_tweet(tweet):
    tweet = re.sub(r"@[A-Za-z0-9]+", "", tweet)  # remove @person
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", "", tweet)  # remove url
    tweet = re.sub("[^a-zA-Z]", " ", tweet)  # only save numbers and characters
    tweet = re.sub("RT", " ", tweet)  # remove re tweet tag
    tweet = tweet.replace("\n", "" "").replace("\r", "")
    tweet = " ".join(tweet.split())
    return tweet


class TwitterApi(object):

    def __init__(self):
        auth = tweepy.OAuthHandler(Config.TWITTER_CONSUMER_KEY, Config.TWITTER_CONSUMER_SECRET_KEY)
        auth.set_access_token(Config.TWITTER_ACCESS_TOKEN, Config.TWITTER_ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(auth)

    def get_topic_of_trends(self, woeid=None, top=None):
        trends = json.loads(json.dumps(self.api.trends_place(id=woeid), indent=1))
        topic_with_volume = {trend["name"].strip("#"): trend["tweet_volume"] for trend in trends[0]["trends"]}
        top_topic = sorted(topic_with_volume, key=lambda k: k[1], reverse=False)
        return top_topic[:top] if top else top_topic

    def get_tweets_of_topic(self, topic, count=10):
        tweets_status = self.api.search(q=topic, count=count)
        tweets = [
            {"topic": topic, "tweet": _clean_tweet(tw.text)} for tw in tweets_status if tw.lang == "en"
        ]
        return tweets


def main():
    twitter_api = TwitterApi()
    topics = twitter_api.get_topic_of_trends(woeid=3534, top=10)
    topics_tweets = []
    for topic in topics:
        topics_tweets.extend(twitter_api.get_tweets_of_topic(topic, count=10))

    schema = StructType([StructField("topic", StringType(), True), StructField("tweet", StringType(), True)])
    spark = SparkSession.builder.master("local").appName("Twitter Sentiment Analysis").getOrCreate()
    spark.createDataFrame(topics_tweets, schema=schema).show(truncate=False)


if __name__ == "__main__":
    main()
