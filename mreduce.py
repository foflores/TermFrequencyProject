from pyspark import SparkContext
from pyspark.conf import SparkConf
from operator import add
from math import log

def to_list(a):
    return [a]
def append(a, b):
    a.append(b)
    return a
def extend(a, b):
    a.extend(b)
    return a

def tfidf(sc, data_path):
    tf_rdd = sc.emptyRDD()
    idf_rdd = sc.emptyRDD()
    tfidf_rdd = sc.emptyRDD()
    data = []

    #reads in file as list of lines
    with open(data_path, 'r') as docs:
        data = docs.readlines()

    total_docs = len(data)

    for line in data:
        #loads data into rdd one document at a time and separates it into individual words
        temp_tf_rdd = sc.parallelize([line]).flatMap(lambda a: a.split(" "))

        #takes the first element(doc_id)
        doc_id = temp_tf_rdd.take(1)

        #filters out elements ' ', '\n', and doc_id
        #then maps each remaining element(term) to key-value pair (term, 1)
        temp_tf_rdd = temp_tf_rdd.filter(lambda a: (len(a) > 0) and (a != '\n') \
                                    and (a != doc_id[0])).map(lambda a: (a,1))
        
        #takes count of the words per line
        total_words = temp_tf_rdd.count()

        #reduceByKey gets a count of each word per line (termcount)
        #then maps each value to (termcount / total_words) which solves the tf portion.
        temp_tf_rdd = temp_tf_rdd.reduceByKey(add).mapValues(lambda a: (a / total_words))

        #takes the keys of temp_tf_rdd and maps each key(term) to key-value pair (term, doc_id)
        temp_idf_rdd = temp_tf_rdd.keys().map(lambda a: (a,doc_id[0]))

        #after each iteration, it unites the current line with idf_rdd which contains the entire file
        idf_rdd = idf_rdd.union(temp_idf_rdd)

        #joins tf and idf rdds and maps them to (term,(doc_id, tf))
        temp_tfidf_rdd = temp_idf_rdd.join(temp_tf_rdd)

        #after each iteration, it unites the current line with tfidf_rdd which contains the entire file
        tfidf_rdd = tfidf_rdd.union(temp_tfidf_rdd)


    #combines all values into a list by its key(term) to end up with key-value pair (term, [doc_id, doc_id2...])
    #then takes count of how many documents each word appears and calculates idf value (term, idf)
    idf_rdd = idf_rdd.combineByKey(to_list, append, extend).map(lambda a: (a[0],log(total_docs/len(a[1]),10)))

    #joins tfidf and idf rdds and maps them to (term, (doc_id, tf), idf)
    tfidf_rdd = tfidf_rdd.join(idf_rdd)

    #does final calculation for tfidf and maps it to (term, (doc_id, tfidf))
    #then combines by key to final output (term, [(doc_id, tfidf), (doc_id2, tfidf2)...])
    tfidf_rdd = tfidf_rdd.mapValues(lambda a: (a[0][0], a[0][1]*a[1])).combineByKey(to_list, append, extend)

    #Test RDDs
    #tf_rdd.saveAsTextFile('data/tf-out')
    #idf_rdd.saveAsTextFile('data/idf-out')
    tfidf_rdd.saveAsTextFile('data/tfidf-out')

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
    tfidf_rdd = tfidf(sc, data_path)

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