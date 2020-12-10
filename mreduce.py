from pyspark import SparkContext
from pyspark.conf import SparkConf
from operator import add
from math import log
from math import sqrt
import os

#functions to be used by combineByKey to combine as list
def c1(a):
    return [a]
def c2(a, b):
    a.append(b)
    return a
def c3(a, b):
    a.extend(b)
    return a

#functions to be used in RDD calculations
def f1(a):
    a = a.split(' ')
    b = []
    for term in a:
        if term == a[0]:
            continue
        b.append((a[0], term))
    return b
def f2(a):
    query = dict(a[1])
    numer = 0
    denom1 = 0
    denom2 = 0
    for pair in a[0]:
        denom1 = denom1 + (pair[1] * pair[1])
        if query.get(pair[0]) != None:
            numer = numer + (pair[1] * query[pair[0]])
    for pair in a[1]:
        denom2 = denom2 + (pair[1] * pair[1])
    return (numer/(sqrt(denom1)*sqrt(denom2)))

#computes tf-idf matrix
def tfidf(sc, data_path):
    #imports data and caches to memory
    tfidf_rdd = sc.textFile(data_path).cache()

    #takes count of total documents
    total_docs = tfidf_rdd.count()

    #maps data into (doc_id, term) and caches to memory
    tfidf_rdd = tfidf_rdd.flatMap(f1).cache()
    
    #filters out empty terms
    tfidf_rdd1 = tfidf_rdd.filter(lambda a: a[1] != '' and a[1] != ' ' and a[1] != '\n').cache()

    #filters out unused terms
    tfidf_rdd2 = tfidf_rdd1.filter(lambda a: a[1].startswith("dis_") or a[1].startswith("gene_")).cache()

    #computes the number of times each word appears in each document (TF numerator)
    tfnumer_rdd = tfidf_rdd2.map(lambda a: (a, 1)).reduceByKey(add)\
        .map(lambda a: (a[0][0], (a[0][1], a[1]))).cache()

    #computes total words per document (TF denominator)
    tfdenom_rdd = tfidf_rdd1.map(lambda a: (a[0], 1)).reduceByKey(add).cache()

    #computes IDF 
    idf_rdd = tfidf_rdd2.distinct().map(lambda a: (a[1], 1)).reduceByKey(add)\
        .mapValues(lambda a: (log((total_docs/a), 10))).cache()

    #joins TF numerator and denominator
    tf_rdd = tfnumer_rdd.join(tfdenom_rdd)

    #computes TF and joins IDF
    tfidf_rdd = tf_rdd.map(lambda a: (a[1][0][0], (a[0], (a[1][0][1]/a[1][1])))).join(idf_rdd)

    #computes TFIDF, combines by key to format (term, [(doc_id1, TFIDF1), (doc_id2, TFIDF2)...]), 
    #and sorts data
    tfidf_rdd = tfidf_rdd.mapValues(lambda a: (a[0][0], (a[0][1]/a[1]))).combineByKey(c1, c2, c3)\
        .sortByKey().cache()

    return tfidf_rdd

#computes relevance scores
def similarity(sc, tfidf_rdd, query):
    #filters for query term
    output = tfidf_rdd.filter(lambda a: a[0] == query).collect()

    #if query term is not found, return empty list
    if len(output) == 0:
        return output
    
    #filters out query term, maps all terms in matrix with query term numbers and caches matrix
    tfidf_rdd = tfidf_rdd.filter(lambda a: a[0] != query).mapValues(lambda a: (a, output[0][1])).cache()

    #calculates similarity, sorts in descending order, and takes first 10
    similarity = tfidf_rdd.mapValues(f2).sortBy(lambda a: a[1], False).take(10)

    return (similarity)

def main():
    #spark configuration options
    conf=SparkConf()\
    .setMaster("local[*]")\
    .setAppName("ttr")\
    .setExecutorEnv("spark.executor.memory","4g")\
    .setExecutorEnv("spark.driver.memory","4g")

    #starts spark context
    sc = SparkContext(conf=conf).getOrCreate()
    
    #data location
    data_path = "data/project2_demo.txt"

    #compute tf-idf matrix
    tfidf_rdd = tfidf(sc, data_path)
    tfidf_rdd.cache()

    #clears screen of spark context startup notifications
    os.system('clear')

    #interface
    done = False
    while not done:
        print("\nEnter a term to see its relevance to all terms in the TF-IDF matrix.")
        query = input("\nTerm: ")
        output = similarity(sc, tfidf_rdd, query)
        if len(output) == 0:
            print("\nTerm not found, try again!")
            continue
        print(f"\nRelevance scores for '{query}': (term, score)\n")
        for a in output:
            print(f"\t{a[0]}, {a[1]}")
        while True:
            answer = input("\nWould you like to try another term? ('y' = yes, 'n' = no): ")
            if answer == 'n':
                done = True
                break
            elif answer == 'y':
                os.system("clear")
                break
            else:
                print("try again!")
    
    #stops spark context
    sc.stop()
    
if __name__ == "__main__":
    main()