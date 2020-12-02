from pyspark import SparkContext
from pyspark.conf import SparkConf
from operator import add
from math import log


def main():
    sc = SparkContext("local", "ttr").getOrCreate()
    tf_rdd = sc.emptyRDD()
    idf_rdd = sc.emptyRDD()
    data = []

    # spark config to avoid manual assignment of mem
    conf=SparkConf()\
    .setExecutorEnv("spark.executor.memory","4g")\
    .setExecutorEnv("spark.driver.memory","4g")

    with open('data\input_test.txt', 'r') as docs:
        data = docs.readlines()

    total_docs = len(data)

    for line in data:
        temp_tf_rdd = sc.parallelize([line]).flatMap(lambda a: a.split(" "))
        tf_rdd_id = temp_tf_rdd.take(1)
        temp_tf_rdd = temp_tf_rdd.filter(lambda a: (len(a) > 0) and (
            a != '\n') and (a != tf_rdd_id[0])).map(lambda a: (a, 1))
        total_words = temp_tf_rdd.count()
        temp_tf_rdd = temp_tf_rdd.reduceByKey(add).mapValues(lambda a:
                                                             (a / total_words))
        tf_rdd = tf_rdd.union(temp_tf_rdd)

        temp_idf_rdd = temp_tf_rdd.keys().map(lambda a: (a, 1))
        idf_rdd = idf_rdd.union(temp_idf_rdd).reduceByKey(add).mapValues(
            lambda a: (log(total_docs / a, 10)))

    idf_rdd = idf_rdd.reduceByKey(add).mapValues(lambda a:
                                                 (log(total_docs / a, 10)))
    tfidf_rdd = idf_rdd.join(tf_rdd).mapValues(lambda x: x[0] * x[1])

    #Test RDDs
    # idf_rdd is the reduced output of the corpus count

    rdd_result =idf_rdd.collect()
    for (word, count) in rdd_result:
        print("%s: %i" % (word, count))

    #tf_rdd.saveAsTextFile('data/tf-out')
    #idf_rdd.saveAsTextFile('data/idf-out')
    tfidf_rdd.saveAsTextFile('data/tfidf-out')

    sc.stop()


if __name__ == "__main__":
    print("running program")
    # main()