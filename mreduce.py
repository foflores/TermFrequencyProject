from pyspark import SparkContext
from pyspark.conf import SparkConf
from operator import add
from math import log
import time

def c1(a):
    return [a]
def c2(a, b):
    a.append(b)
    return a
def c3(a, b):
    a.extend(b)
    return a

def f1(a):
    a = a.split(' ')
    b = []
    for term in a:
        if term == a[0]:
            continue
        b.append((term, a[0]))
    return b

#previous avg run time: 20 seconds
#current avg run time: 3 seconds
def tfidf(sc, data_path):
    #imports data
    tfidf_rdd = sc.textFile(data_path)
    tfidf_rdd.cache()

    #takes count of total documents
    total_docs = tfidf_rdd.count()

    #maps data into (term, doc_id) and filters empty characters
    tfidf_rdd = tfidf_rdd.flatMap(f1).filter(lambda a: a[0] != ' ' and a[0] != '\n' and a[0] != '')
    tfidf_rdd.cache()

    #combines data into (term, [doc_id, doc_id2...]) and maps it to (term, idf)
    idf_rdd = tfidf_rdd.combineByKey(c1, c2, c3)
    idf_rdd = idf_rdd.map(lambda a: (a[0], log(total_docs/len(list(set(a[1])))), 10))
    idf_rdd.cache()

    #maps data to ((term, doc_id), 1) and reduces it to count term frequency
    #per document
    tf_rdd = tfidf_rdd.map(lambda a: ((a[0], a[1]), 1))
    tf_rdd = tf_rdd.reduceByKey(add)
    tf_rdd.cache()

    #maps data to (doc_id, 1) and reduces it to count words per document
    temp_tf_rdd = tfidf_rdd.map(lambda a: (a[1],1))
    temp_tf_rdd = temp_tf_rdd.reduceByKey(add)
    temp_tf_rdd.cache()

    #maps data to (doc_id,(word, term)) and joins it with temp_tf_rdd
    #contains word count per document
    tf_rdd = tf_rdd.map(lambda a: (a[0][1],(a[0][0], a[1])))
    tf_rdd = tf_rdd.join(temp_tf_rdd)

    #maps data to (word,(doc_id, tf))
    tf_rdd = tf_rdd.map(lambda a: (a[1][0][0], (a[0],a[1][0][1]/a[1][1])))
    tf_rdd.cache()

    #joins tf_rdd and idf_rdd
    tfidf_rdd = tf_rdd.join(idf_rdd)

    #maps data to (word, (doc_id, tfidf)) and then combines it to
    #(word, [(doc_id, tfidf), (doc_id2, tfidf2)...])
    tfidf_rdd = tfidf_rdd.mapValues(lambda a: (a[0][0], a[0][1]*a[1]))
    tfidf_rdd = tfidf_rdd.combineByKey(c1, c2, c3)

    #Test RDDs
    #tf_rdd.saveAsTextFile('data/tf-out')
    #idf_rdd.saveAsTextFile('data/idf-out')
    #tfidf_rdd.saveAsTextFile('data/tfidf-out')

    return tfidf_rdd

def similarity(sc, tfidf_rdd, query):
    #placeholder for your function
    return [1,2,3]

def main():
    #spark config options
    conf=SparkConf()\
    .setMaster("local[*]")\
    .setAppName("ttr")\
    .setExecutorEnv("spark.executor.memory","4g")\
    .setExecutorEnv("spark.driver.memory","4g")
    sc = SparkContext(conf=conf).getOrCreate()
    
    data_path = "data/input_test.txt"

    #lines 95 and 97 are just to test run time, they can be deleted
    start_time = time.time()
    tfidf_rdd = tfidf(sc, data_path)
    print ("\nExecution time: %s seconds" % (time.time()-start_time))

    #Interface is complete, you can uncomment it once your function is done
    """
    done = False
    while not done:
        print("\nEnter a term to see its similarity to all terms in the TF-IDF matrix.")
        query = input("\nTerm: ")
        output = similarity(sc, tfidf_rdd, query)
        output = [2]
        if len(output) == 0:
            print("\nTerm not found, try again!")
            continue
        print(f"\nSimilarity scores for {query}: (term, score)")
        for a in output:
            print(a)
        while True:
            print("\nWould you like to try another term?")
            answer = input("Enter 'y' for yes or 'n' for no: ")
            if answer == 'n':
                done = True
                break
            elif answer == 'y':
                break
            else:
                print("try again!")
    """

    sc.stop()

if __name__ == "__main__":
    main()