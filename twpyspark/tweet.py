import json
import re

import tweepy
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from twpyspark.settings import Config
from twpyspark.visualize import visualize


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def _clean_tweet(tweet):
    tweet = decontracted(tweet)
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
        tweets_status = self.api.search(q=topic + " -filter:retweets", count=count, lang='en')
        if len(tweets_status) != count:
            tweets_status = self.api.search(q=topic, count=count, lang='en')
        tweets = [{"topic": topic, "tweet": _clean_tweet(tw.text)} for tw in tweets_status]
        return tweets


def get_sentimment(sentence: str, analyzer: SentimentIntensityAnalyzer):
    """
    >>> get_sentimment(sentence='I am a smart boy!', analyzer=SentimentIntensityAnalyzer())
    1

    >>> get_sentimment(sentence='I am a bad boy!', analyzer=SentimentIntensityAnalyzer())
    -1

    >>> get_sentimment(sentence='I am a boy!', analyzer=SentimentIntensityAnalyzer())
    0
    """
    score = analyzer.polarity_scores(sentence)
    compound = score['compound']
    if compound >= 0.1:
        return 1  # positive
    elif (compound > -0.1) and (compound < 0.1):
        return 0  # nuture
    else:
        return -1  # negative


def main():
    twitter_api = TwitterApi()
    topics = twitter_api.get_topic_of_trends(woeid=Config.WOEID, top=Config.TWEETS_TOP_TOPIC)  # 3534 is Canada
    topics_tweets = []
    for topic in topics:
        topics_tweets.extend(twitter_api.get_tweets_of_topic(topic, count=Config.TWEETS_COUNT))

    schema = StructType([StructField("topic", StringType(), True), StructField("tweet", StringType(), True)])
    spark = SparkSession.builder.master("local").appName("Twitter Sentiment Analysis").getOrCreate()
    df = spark.createDataFrame(topics_tweets, schema=schema)
    # sentiment
    analyzer = SentimentIntensityAnalyzer()
    sentiment_udf = udf(lambda tweet: get_sentimment(tweet, analyzer), IntegerType())
    sentiment_df = df.withColumn("sentiment", sentiment_udf(df.tweet)).orderBy(df.topic.desc())
    sentiment_df.show(n=50, truncate=False)

    # each topic's sentiment count
    from pyspark.sql import functions as F
    sentiment_count_by_topic = sentiment_df \
        .groupBy("topic") \
        .pivot("sentiment") \
        .agg(F.count("sentiment")) \
        .na.fill(0)
    sentiment_count_by_topic = sentiment_count_by_topic \
        .select(F.col("topic"),
                F.col("-1").alias("negative"),
                F.col("0").alias("neutral"),
                F.col("1").alias("positive"))
    sentiment_count_by_topic.show()

    # visualization
    sentiment_count_by_topic_dict = {r['topic']: [r['negative'], r['neutral'], r['positive']]
                                     for r in sentiment_count_by_topic.collect()}
    category_names = sentiment_count_by_topic.columns[1:]
    visualize(sentiment_count_by_topic_dict, category_names)


if __name__ == "__main__":
    main()
