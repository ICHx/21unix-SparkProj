from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
from operator import add

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import Word2Vec

DEBUG = True


def main():
    print(f"Hello world, decoding {fname}")
    spark = SparkSession.builder\
        .appName("MyProj")\
        .getOrCreate()

    df = spark.read.text(fname, wholetext=False)

    tokenizer = Tokenizer(inputCol="value", outputCol="token")

    tokenized = tokenizer.transform(df)

    word2Vec = Word2Vec(vectorSize=3, minCount=0,
                        inputCol="token", outputCol="vector")

    model = word2Vec.fit(tokenized)
    model.write().overwrite().save("hdfs://zuk:9000/history_model")

    result = model.transform(tokenized)

    # extract columns and plot
    wds = result.select("token").collect()
    vecs = result.select("vector").collect()

    plotVecs(wds, vecs, result.count())

    # clean
    result.show()
    spark.stop()
    return


def plotVecs(wds, vecs, count):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(count):
        label = str(wds[i][0])
        coords = vecs[i][0].toArray()
        ax.scatter(*coords)
        ax.text(*coords, label)

    plt.show()
    oname= fname.replace("/","").replace(":","")
    # oname= fname.split("/")[-1]
    plt.savefig(f"/opt/spark/myProj/plots/out-{oname}.png")
    pass


if __name__ == '__main__':
    global fname
    print(sys.argv)
    if len(sys.argv) < 2:
        fname = "hdfs://zuk:9000/history"
    else:
        fname = sys.argv[-1]

    main()
