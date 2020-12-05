from pyspark import SparkContext
from pyspark.conf import SparkConf
from operator import add
from math import log

def tfidf(sc, data_path):
    tf_rdd = sc.emptyRDD()
    idf_rdd = sc.emptyRDD()
    data = []

    #reads in file as list of lines
    with open(data_path, 'r') as docs:
        data = docs.readlines()

    total_docs = len(data)

    for line in data:
        #loads data into rdd one document at a time and separates it into individual words
        temp_tf_rdd = sc.parallelize([line]).flatMap(lambda a: a.split(" "))

        #takes the first element of the line (i think the first element is a sort of id, so im 
        #separating it as of now b/c i am not sure what to do with it)
        tf_rdd_id = temp_tf_rdd.take(1)

        #filters out elements such as ' ', '\n', and the first id element
        #then maps each remaining element(a) to key-value pair (a, 1)
        temp_tf_rdd = temp_tf_rdd.filter(lambda a: (len(a) > 0) and (
            a != '\n') and (a != tf_rdd_id[0])).map(lambda a: (a, 1))
        
        #takes count of the words in the line
        total_words = temp_tf_rdd.count()

        #reduceByKey adds all values with the same key to get a count of each word per line
        #then maps each value to (a / total_words) which solves the tf portion.
        temp_tf_rdd = temp_tf_rdd.reduceByKey(add).mapValues(lambda a:
                                                             (a / total_words))
        #after each iteration, it unites the current line with tf_rdd which contains the entire file
        tf_rdd = tf_rdd.union(temp_tf_rdd)

        #takes only the keys from temp_tf_rdd and creates a new RDD
        #then maps each of those keys(a) to a key-value pair (a,1)
        temp_idf_rdd = temp_tf_rdd.keys().map(lambda a: (a, 1))

        #after each iteration, it unites the current line with idf_rdd which contains the entire file
        idf_rdd = idf_rdd.union(temp_idf_rdd)

    #reducebyKey counts total number of documents each word appears in
    #then maps each value to to log(total_docs / a, 10) which solves the idf portion
    idf_rdd = idf_rdd.reduceByKey(add).mapValues(lambda a:
                                                 (log(total_docs / a, 10)))

    #joins tf_rdd and idf_rdd by key and maps each value(x[0], x[1]) to (x[0] * x[1])
    #tfidf_rdd contains final tfidf calucation for each word per document.
    tfidf_rdd = tf_rdd.join(idf_rdd).mapValues(lambda x: x[0] * x[1])

    return tfidf_rdd

def similarity(sc, tfid_rdd, query):
    return [1,2,3]

def main():
    #spark config options
    conf=SparkConf()\
    .setMaster("local")\
    .setAppName("ttr")\
    .setExecutorEnv("spark.executor.memory","4g")\
    .setExecutorEnv("spark.driver.memory","4g")
    sc = SparkContext(conf=conf).getOrCreate()
    
    data_path = "data/input_test.txt"
    tfidf_rdd = tfidf(sc, data_path)

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
    #Test RDDs
    #tf_rdd.saveAsTextFile('data/tf-out')
    #idf_rdd.saveAsTextFile('data/idf-out')
    #tfidf_rdd.saveAsTextFile('data/tfidf-out')

    sc.stop()

if __name__ == "__main__":
    main()