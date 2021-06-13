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

DEBUG = True

result = None


def main(fname="hdfs://zuk:9000/dataset/twitter/Tweets.csv"):
    global result
    oname = fname.replace("/", "").replace(":", "")

    print(f"Hello world, decoding {fname}")
    print("Last changed: 14:16")
    spark = SparkSession.builder\
        .appName("MyProj")\
        .master("spark://virtual01-virtualbox:7077")\
        .config("spark.sql.repl.eagerEval.enabled", True)\
        .getOrCreate()
        # .master("local[*]")\

    # ! read data
    df = spark.read.option("header", True).csv(fname)\
        .select("airline_sentiment", "text")

    rter = RegexTokenizer(inputCol="text", outputCol="token",
                          pattern="[\\W]+", minTokenLength=1)
    tokenized = rter.transform(df.na.fill(''))

    remover = StopWordsRemover(inputCol="token", outputCol="rtoken")  # ok
    removed = remover.transform(tokenized)  # ok

    # !seperate data
    posDF = removed.filter("airline_sentiment = \"positive\" OR airline_sentiment = \"neutral\"")

    negDF = removed.filter(removed.airline_sentiment == "negative")

    # ! word2vec
    word2Vec = Word2Vec(vectorSize=3, minCount=3,
                        inputCol="rtoken", outputCol="vector")

    posModel = word2Vec.fit(posDF)
    negModel = word2Vec.fit(negDF)

    # !transform data
    posResult = posModel.transform(posDF)
    negResult = negModel.transform(negDF)

    # ! extract columns and plot

    wds1 = posResult.select("rtoken")\
        .collect()
    vecs1 = posResult.select("vector")\
        .collect()

    wds2 = negResult.select("rtoken")\
        .collect()
    vecs2 = negResult.select("vector")\
        .collect()

    cnt1 = len(vecs1)
    cnt2 = len(vecs2)
    plotVecs(oname+"pos1", wds1, vecs1, cnt1)
    plotVecs(oname+"neg1", wds2, vecs2, cnt2)

    # ! clean
    print(f"count= +{cnt1} -{cnt2}")
    spark.stop()
    return


def plotVecs(oname, wds, vecs, count):
    plt.ioff()
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for i in range(count):
        coords = vecs[i][0].toArray()
        ax.scatter(*coords)
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
