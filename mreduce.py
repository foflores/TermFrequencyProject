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
        b.append((term, a[0]))
    return b
def f2(a):
    term = dict(a[0])
    query = dict(a[1])
    output = 0
    for id in term:
        if query.get(id) != None:
            output = output + (term[id] * query[id])
    return output
def f3(a):
    term = a[0]
    query = a[1]
    num1 = 0
    num2 = 0
    for pair in term:
        num1 = num1 + (pair[1] * pair[1])
    for pair in query:
        num2 = num2 + (pair[1] * pair[1])
    return (sqrt(num1) * sqrt(num2))

#computes tf-idf matrix
def tfidf(sc, data_path):
    #imports data and caches to memory
    tfidf_rdd = sc.textFile(data_path)
    tfidf_rdd.cache()

    #takes count of total documents
    total_docs = tfidf_rdd.count()

    #maps data into (term, doc_id), filters empty characters, and caches to memory
    tfidf_rdd = tfidf_rdd.flatMap(f1).filter(lambda a: a[0] != ' ' and a[0] != '\n' and a[0] != '')
    tfidf_rdd.cache()

    #combines data into (term, [doc_id, doc_id2...]), maps it to (term, idf), and caches to memory
    idf_rdd = tfidf_rdd.combineByKey(c1, c2, c3)
    idf_rdd = idf_rdd.map(lambda a: (a[0], log(total_docs/len(set(a[1])), 10)))
    idf_rdd.cache()

    #maps data to ((term, doc_id), 1), reduces it to count term frequency, and caches to memory
    tf_rdd = tfidf_rdd.map(lambda a: ((a[0], a[1]), 1))
    tf_rdd = tf_rdd.reduceByKey(add)
    tf_rdd.cache()

    #maps data to (doc_id, 1), reduces it to count words per document, and caches to memory
    temp_tf_rdd = tfidf_rdd.map(lambda a: (a[1],1))
    temp_tf_rdd = temp_tf_rdd.reduceByKey(add)
    temp_tf_rdd.cache()

    #maps data to (doc_id,(word, term)) and joins it with temp_tf_rdd
    #which contains word count per document
    tf_rdd = tf_rdd.map(lambda a: (a[0][1],(a[0][0], a[1])))
    tf_rdd = tf_rdd.join(temp_tf_rdd)

    #maps data to (word,(doc_id, tf))
    tf_rdd = tf_rdd.map(lambda a: (a[1][0][0], (a[0],a[1][0][1]/a[1][1])))
    tf_rdd.cache()

    #joins tf_rdd and idf_rdd
    tfidf_rdd = tf_rdd.join(idf_rdd)

    #maps data to (word, (doc_id, tfidf)), combines it to
    #(word, [(doc_id, tfidf), (doc_id2, tfidf2)...]), and caches to memory
    tfidf_rdd = tfidf_rdd.mapValues(lambda a: (a[0][0], a[0][1]*a[1])).combineByKey(c1, c2, c3)
    tfidf_rdd.cache()

    return tfidf_rdd

#computes relevance scores
def similarity(sc, tfidf_rdd, query):
    #checks query term against matrix to see if it exists in data
    output = tfidf_rdd.filter(lambda a: a[0] == query).collect()
    if len(output) == 0:
        return output
    
    #filters out query term from matrix and caches matrix
    tfidf_rdd = tfidf_rdd.filter(lambda a: a[0] != query).mapValues(lambda a: (a, output[0][1]))
    tfidf_rdd.cache()

    #calculates numerator of similarity function and caches to memory
    numerator_rdd = tfidf_rdd.mapValues(f2)
    numerator_rdd.cache()

    #calculates denominator of similarity function and caches to memory
    denominator_rdd = tfidf_rdd.mapValues(f3)
    denominator_rdd.cache()

    #joins numerator and denominator rdds, computes scores, and sorts in descending order
    similarity_rdd = numerator_rdd.join(denominator_rdd)
    similarity_rdd = similarity_rdd.mapValues(lambda a: a[0]/a[1]).sortBy(lambda a: a[1], False)

    #filters out any scores of 0
    similarity_rdd = similarity_rdd.filter(lambda a: a[1] != 0)

    return (similarity_rdd.collect())

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
    data_path = "data/input_test.txt"

    #compute tf-idf matrix
    tfidf_rdd = tfidf(sc, data_path)

    #clears screen of spark context startup notifications
    os.system('clear')

    #query interface
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
            answer = input("Would you like to try another term? ('y' = yes, 'n' = no): ")
            if answer == 'n':
                done = True
                break
            elif answer == 'y':
                os.system('clear')
                break
            else:
                print("try again!")
    
    #stops spark context
    sc.stop()
    
if __name__ == "__main__":
    main()