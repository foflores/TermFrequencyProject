from pyspark import SparkContext
from operator import add

def main():
    sc = SparkContext("local", "ttr").getOrCreate()
    tf_rdd = sc.emptyRDD()
    idf_rdd = sc.emptyRDD()
    data = []

    with open('data/input1.txt', 'r') as docs:
        data = docs.readlines()
    
    total_docs = len(data)

    for line in data:
        tf_rdd = sc.parallelize([line]).flatMap(lambda a: a.split(" "))
        tf_rdd_id = tf_rdd.take(1)
        tf_rdd = tf_rdd.filter(lambda a:(len(a)>0) and (a!='\n') and (a!=tf_rdd_id[0])).map(lambda a:(a,1))
        total_words = tf_rdd.count()
        tf_rdd = tf_rdd.reduceByKey(add).mapValues(lambda a: (a/total_words))

        idf_rdd = tf_rdd.keys()
        idf_rdd = idf_rdd.map(lambda a: (a,1))

    idf_rdd = idf_rdd.reduceByKey(add)

    #Test RDDs
    #print(tf_rdd.collect())
    #print(idf_rdd.collect())
    


if __name__ == "__main__":
    main()