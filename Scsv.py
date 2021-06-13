from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
from operator import add

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StopWordsRemover

import time

DEBUG = True
MASTER= "spark://virtual01-virtualbox:7077"
# MASTER= "local[*]"


def main(fname="hdfs://zuk:9000/dataset/twitter/Tweets.csv"):
    t1 = time.time()
    oname = fname.replace("/", "").replace(":", "")

    print(f"Hello world, decoding {fname}")
    print("Last changed: 19:16")
    spark = SparkSession.builder\
        .appName("MyProj")\
        .master(MASTER)\
        .config("spark.sql.repl.eagerEval.enabled", True)\
        .getOrCreate()

    # ! read data
    df = spark.read.option("header", True).csv(fname)\
        .select("airline_sentiment", "text")

    rter = RegexTokenizer(inputCol="text", outputCol="token",
                          pattern="[\\W]+", minTokenLength=1)
    tokenized = rter.transform(df.na.fill(''))

    remover = StopWordsRemover(inputCol="token", outputCol="rtoken")  # ok
    removed = remover.transform(tokenized)  # ok

    # ! word2vec
    word2Vec = Word2Vec(vectorSize=3, minCount=3,
                        inputCol="rtoken", outputCol="vector")

    Model = word2Vec.fit(removed)

    Result = Model.transform(removed)

    # ! extract columns and plot

    #? positive 
    wds1 = Result.select("rtoken")\
        .filter("airline_sentiment = \"positive\" OR airline_sentiment = \"neutral\"")\
        .collect()
        
    vecs1 = Result.select("vector")\
        .filter("airline_sentiment = \"positive\" OR airline_sentiment = \"neutral\"")\
        .collect()

    #? negative
    wds2 = Result.select("rtoken")\
        .filter(removed.airline_sentiment == "negative")\
        .collect()
    vecs2 = Result.select("vector")\
        .filter(removed.airline_sentiment == "negative")\
        .collect()


    cnt1 = len(vecs1)
    cnt2 = len(vecs2)
    plotVecs(oname+"pos1", wds1, vecs1, cnt1, "red")
    plotVecs(oname+"neg1", wds2, vecs2, cnt2, "blue")
    
    plotBothVecs(oname+"both1", wds2, vecs2, cnt2, "blue",  wds1, vecs1, cnt1, "red")

    # ! clean
    print(f"count= +{cnt1} -{cnt2}")
    spark.stop()
    
    t2 = time.time() - t1
    
    print(f"Using {MASTER}, used {t2}s.")
    return


def plotVecs(oname, wds, vecs, count, color):
    plt.ioff()
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for i in range(count):
        coords = vecs[i][0].toArray()
        ax.scatter(*coords, c=color)
        pass

    # too much too slow and crowded, disabled
    # label = str(wds[i][0])
    # ax.text(*coords, label)

    plt.savefig(f"/opt/spark/myProj/plots/out-{oname}.png")
    return fig  # in jupyter


def plotBothVecs(oname, wds, vecs, count, color, wds2, vecs2, count2, color2):
    plt.ioff()
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for i in range(count):
        coords = vecs[i][0].toArray()
        ax.scatter(*coords, c=color)
        pass
    
    for i in range(count2):
        coords = vecs2[i][0].toArray()
        ax.scatter(*coords, c=color)
        pass

    # too much too slow and crowded, disabled
    # label = str(wds[i][0])
    # ax.text(*coords, label)

    plt.savefig(f"/opt/spark/myProj/plots/out-{oname}.png")
    return fig  # in jupyter


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) < 2:
        main()
    else:
        fname = sys.argv[1]
        main(fname)
