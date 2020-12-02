from pyspark import SparkContext
from operator import add
from math import log

def main():
    sc = SparkContext("local", "ttr").getOrCreate()
    tf_rdd = sc.emptyRDD()
    idf_rdd = sc.emptyRDD()
    data = []

    with open('data/input1.txt', 'r') as docs:
        data = docs.readlines()
    
    total_docs = len(data)

    for line in data:
        temp_tf_rdd = sc.parallelize([line]).flatMap(lambda a: a.split(" "))
        tf_rdd_id = temp_tf_rdd.take(1)
        temp_tf_rdd = temp_tf_rdd.filter(lambda a:(len(a)>0) and (a!='\n') and (a!=tf_rdd_id[0])).map(lambda a:(a,1))
        total_words = temp_tf_rdd.count()
        temp_tf_rdd = temp_tf_rdd.reduceByKey(add).mapValues(lambda a: (a/total_words))
        tf_rdd = tf_rdd.union(temp_tf_rdd)

        temp_idf_rdd = temp_tf_rdd.keys().map(lambda a: (a,1))
        idf_rdd = idf_rdd.union(temp_idf_rdd).reduceByKey(add).mapValues(lambda a: (log(total_docs/a, 10)))
        
    idf_rdd = idf_rdd.reduceByKey(add).mapValues(lambda a: (log(total_docs/a, 10)))
    tfidf_rdd = idf_rdd.join(tf_rdd).mapValues(lambda x: x[0]*x[1])

    #Test RDDs
    #tf_rdd.saveAsTextFile('data/tf-out')
    #idf_rdd.saveAsTextFile('data/idf-out')
    tfidf_rdd.saveAsTextFile('data/tfidf-out')


if __name__ == "__main__":
    main()