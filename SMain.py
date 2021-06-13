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
fig = None
result= None


def main(fname="/history"):
    global result
    
    print(f"Hello world, decoding {fname}")
    print("Last changed: 13:13")
    spark = SparkSession.builder\
        .appName("MyProj")\
        .master("local[*]")\
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

    cnt = result.count()
    plotVecs(fname, wds, vecs, cnt)

    # clean
    result.show()
    print(f"count= {cnt}")
    spark.stop()
    return


def plotVecs(fname, wds, vecs, count):
    global fig
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    for i in range(count):
        label = str(wds[i][0])
        coords = vecs[i][0].toArray()
        ax.scatter(*coords)
        
        if(i%10 ==0):
            ax.text(*coords, label)
        pass

    oname = fname.replace("/", "").replace(":", "")
    # oname= fname.split("/")[-1]
    plt.savefig(f"/opt/spark/myProj/plots/out-{oname}.png")

    # plt.show()
    
    
    pass


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) < 2:
        main("hdfs://zuk:9000/history")
    else:
        fname = sys.argv[1]
        main(fname)
