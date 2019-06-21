import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import csv
import pprint
from pymongo import MongoClient

DATA_DIR = "./caption-contest-data/contests/summaries/"
INFO_DIR = "./caption-contest-data/contests/info/adaptive/"

STOP_WORDS = {'about',
 'again',
 'all',
 'alway',
 'am',
 'an',
 'and',
 'are',
 'as',
 'ask',
 'at',
 'back',
 'be',
 'been',
 'better',
 'but',
 'by',
 'call',
 'can',
 'come',
 'could',
 'day',
 'did',
 'didn',
 'do',
 'don',
 'down',
 'feel',
 'first',
 'for',
 'from',
 'get',
 'give',
 'go',
 'good',
 'got',
 'had',
 'has',
 'have',
 'he',
 'here',
 'him',
 'his',
 'how',
 'if',
 'in',
 'is',
 'it',
 'just',
 'keep',
 'know',
 'last',
 'let',
 'like',
 'littl',
 'll',
 'look',
 'make',
 'me',
 'more',
 'my',
 'need',
 'never',
 'new',
 'next',
 'no',
 'not',
 'now',
 'of',
 'off',
 'on',
 'one',
 'onli',
 'or',
 'our',
 'out',
 'over',
 'realli',
 'right',
 'said',
 'say',
 'see',
 'should',
 'so',
 'some',
 'still',
 'sure',
 'take',
 'tell',
 'than',
 'that',
 'the',
 'them',
 'there',
 'these',
 'they',
 'thing',
 'think',
 'this',
 'thought',
 'time',
 'to',
 'told',
 'too',
 'tri',
 'up',
 'us',
 'use',
 'want',
 'was',
 'way',
 'we',
 'well',
 'were',
 'what',
 'when',
 'whi',
 'who',
 'will',
 'with',
 'would',
 'yes',
 'you',
 'your'}

def csv_to_mongo(contest, algo):
    """
    Imports CSV files from the nextml repo to MongoDB.

    contest: three-digit contest number in int format.
    algo: weighting scheme as it appear in the file name,
    in str format. This is "KLUCB" for all recent contests.

    Returns nothing.
    """
    filename = str(contest)+"_summary_"+algo+".csv"
    
    with open(DATA_DIR+filename, "r") as f:
        reader = csv.DictReader(f)
        client = MongoClient()
        db = client.captions
        header = [
                  'rank', 'funny', 'somewhat_funny', 'unfunny', 'count', 'score',
                  'precision', 'contest', 'caption'
                 ]
        
        for csvrow in reader:
            row = {}
            for column in header:
                data = csvrow[column]
                
                if column=='caption':
                    pass
                elif column=='score' or column=='precision':
                    data = float(data)
                else:
                    data = int(data)
                    
                row[column] = data
                
            db.contest.insert_one(row)
        
def regex_caption_search(regex):
    """
    Regex searches all captions stored in MongoDB
    and prints the results. Returns nothing.
    """
    client = MongoClient()
    db = client.captions
    cursor = db.contest.find({'caption': {'$regex': regex}}, {'_id': 0})
    pp = pprint.PrettyPrinter()
    pp.pprint(list(cursor))

def load_captions():
    """
    Loads caption contest data from MongoDB, skipping contests
    with duplicate data.

    No parameters.
    Returns a pandas dataframe.
    """
    client = MongoClient()
    db = client.captions
    cursor = db.contest.find({}, {'_id':0})
    cdb = pd.DataFrame(list(cursor))
    dupe_filter = (cdb['contest'] != 644) & (cdb['contest'] != 647) & (cdb['contest'] != 656)
    cdb = cdb[dupe_filter]
    return cdb

def show_cartoon(contest):
    """
    Displays a single cartoon (assuming directory structure).

    contest: three-digit contest number, in int format.
    Returns nothing.
    """
    image = mpimg.imread(INFO_DIR+str(contest)+"/"+str(contest)+".jpg")
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('off')

def show_labeled_cartoons(df, label):
    """
    Displays cartoons associated with a particular topic label.

    df: target dataframe. Must have 'label' and 'contest' columns.
    label: label of requested cartoons. Can be string or cluster int.

    Returns nothing.
    """
    contest_ids = df[df['label']==label]['contest']
    for c_id in contest_ids:
        print(f"Cartoon #{c_id}:")
        show_cartoon(c_id)

def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Displays words associated with the topics in a trained
    topic model.

    model: trained sklearn-style topic model.
    feature_names: features ('words') in vectorized data.
    no_top_words: number of words shown per topic.
    topic_names: list of labels for topics.

    Returns nothing.
    """
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))