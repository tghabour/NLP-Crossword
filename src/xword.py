
#Import dependencies
import re
import os
import nltk
import json
import string
import requests
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from nltk.corpus import stopwords
from textblob import TextBlob, Word
from wordsegment import load, segment
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

load()
stop_words = stopwords.words('english')

##########################################################################

def get_jsons(base_path):
    """
    Retrieves individual puzzle jsons from local directory for analysis.
    """

    entries = os.listdir(base_path)
    all_files = []
    for entry in entries:
        path = os.path.join(base_path, entry)
        if os.path.isdir(path):
            all_files = all_files + get_jsons(path)
        elif path.endswith('.json'):
            all_files.append(path)
        else:
            pass

    return all_files

##########################################################################

def fillin_pct(clues_list):
    """
    Returns proportion of a given puzzles clues that are "fill in the blank."
    """

    counter = 0
    for clue in clues_list:
         if clue.find('___') != -1:
            counter += 1
    return counter/len(clues_list)

##########################################################################

def ques_pct(clues_list):
    """
    Returns the proportion of a given puzzles clues that are questions
    (i.e., puns/word play).
    """

    counter = 0
    for clue in clues_list:
         if clue.find('?') != -1:
            counter += 1
    return counter/len(clues_list)

##########################################################################

def quotes_pct(clues_list):
    """
    Returns proportion of a given puzzles clues that are include quotation marks
    (typically literary references or book, movie, song titles).
    """

    counter = 0
    for clue in clues_list:
         if clue.find('\"') != -1:
            counter += 1
    return counter/len(clues_list)

##########################################################################

def self_ref(clues_list):
    """
    Returns proportion of a given puzzles clues that self-referential (i.e.,
    refer to other Down or Across clues within the puzzle).
    """

    counter = 0
    for clue in clues_list:
        if clue.find('Across') != -1:
            counter += 1
        if clue.find('Down') != -1:
            counter += 1
    return counter/len(clues_list)

##########################################################################

def segment_list(words_list):
    """
    Simple function to segment answers provided as contiguous characters into
    separate words.
    """
    words = []
    for chars in words_list:
        words.append(segment(chars))

    all_words = [word for words in words for word in words]

    return all_words

##########################################################################

def clean_sentence(sentence):
    """
    Cleaning function that eliminates digits/punctuation and tokenizes sentences
    (specifically puzzle clues).
    """
    clean = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
    clean = re.sub('\w*\d\w*', ' ', clean)
    tokd = nltk.word_tokenize(clean)
    return tokd

##########################################################################

def clean_list(sentence_list):
    """
    Applies clean_sentence function to a list of sentences and returns
    flattened output list.
    """
    sentences = []
    for sent in sentence_list:
        sentences.append(clean_sentence(sent))

    all_sentences = [sentence for sentences in sentences for sentence in sentences]

    return all_sentences

##########################################################################

def process_text(raw_text_list):
    """
    Processes input text (list of sentences) cleaning (see cleaning functions),
    segmenting, and lemmatizing.
    """
    processed = clean_list(raw_text_list)
    processed = segment_list(processed)
    processed = [word for word in processed if word not in stop_words]

    blob = TextBlob(' '.join(processed))

    lemmed = [Word(x).lemmatize() for x in list(blob.words)]
    lemmed = [x for x in lemmed if len(x)>2]

    return lemmed

##########################################################################

def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Displays top words associated with topics resulting from dimensionality
    reduction techniques (namely, LSA and NMF).
    """
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] \
              for i in topic.argsort()[:-no_top_words - 1:-1]]))

##########################################################################

def date_range(start_date, end_date):

    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)

##########################################################################

def reduce_col(input_string):
    """
    Cleans up clues and answers columns in corpus dataframe after importing from csv.
    """
    input_string = input_string.split("\'")
    input_string = [x for x in input_string if x != ', ']
    input_string = [x for x in input_string if x != '[']
    input_string = [x for x in input_string if x != ']']

    word_list = ' '.join(input_string)

    return word_list

##########################################################################

def get_puzzles(dt_range):
    """
    Scrapes puzzles pertaining to desired input daterange and returns result
    as list of dictionaries.
    """
    puzzles = []
    for date in dt_range:
        year = f'{date.year:04}'
        month = f'{date.month:02}'
        day = f'{date.day:02}'

        url = ("https://raw.githubusercontent.com/doshea/nyt_crosswords/master/" +
               year + '/' + month + '/' + day + ".json")

        response = requests.get(url)

        if response.status_code == 200:
            page = response.text
        else:
            pass

        puzzles.append(json.loads(page))

    return puzzles

##########################################################################

def number_answers(puzz):

    if not puzz.answers.numbered:
        for idx, clue in enumerate(puzz.clues.across):
            num = clue[0:clue.find('.')+1]
            ans = f"{num} {puzz.answers.across[idx]}"
            puzz.answers.across[idx] = ans

        for idx, clue in enumerate(puzz.clues.down):
            num = clue[0:clue.find('.')+1]
            ans = f"{num} {puzz.answers.down[idx]}"
            puzz.answers.down[idx] = ans

    puzz.answers.numbered = True

##########################################################################

def heat_map(puzzle_list, days = 'all'):
    """
    Converts list of puzzle grids into heatmap showing locations of black blocks.
    """

    if days == 'all':
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        num_plots = 7
    else:
        num_plots = len(days)

    max_size = 23
    #plot size =
    sum_ = np.zeros((max_size,max_size))

    for puzz in puzzle_list:
        p = puzzle()
        p.parse_puzzle(puzz)

        bl_list = []
        if p.dow in days:
            grid = [re.sub("[^\.]", "0", char) for char in p.grid]
            grid = [char.replace('.', "1") for char in grid]

            if abs(p.rows - p.cols) < 1:
                for char in grid:
                    bl_list.append(re.sub("[A-Z]", "0", char))

                bl_list = [x[0] for x in bl_list]
                bl_list = np.reshape(bl_list, (p.rows, p.cols))
                bl_list = np.pad(bl_list, int((max_size-p.rows)/2), mode='constant')
                bl_list = bl_list.astype(int)
            else:
                bl_list = np.zeros((max_size,max_size))

            sum_ = np.add(sum_, bl_list)

    fig = plt.figure(figsize=(7, 5))
    fig = sns.heatmap(sum_[1:22, 1:22], xticklabels=2, yticklabels=2, cmap='afmhot_r');

    pass

##########################################################################

def show_cm(model, model_data, colormap,  title = ''):
    """
    Generates confusion matrix (w/r/t test set), for a given classification model.
    """

    classes = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    labels = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

    X_train, X_test, y_train, y_test = model_data

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    model_cm = confusion_matrix(y_test, model.predict(X_test), labels = classes)

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap = plt.get_cmap(colormap)
    new_cmap = truncate_colormap(cmap, 0.0, .9)

    plt.figure(figsize=(7, 7))
    plt.title(title, fontsize= 17)
    plt.tick_params(axis='both', which='major')

    ticks = labels

    sns.set(font_scale=1.3)
    sns.heatmap(model_cm, annot = True, cmap=new_cmap, fmt='g', center = 1,
                cbar = False, xticklabels = ticks, yticklabels = ticks, annot_kws={"size": 14})

    plt.yticks(va="center")
    plt.ylabel("ACTUAL", fontsize = 14, labelpad = 12)
    plt.xlabel("PREDICTED", fontsize = 14, labelpad = 12)

    plt.show()

    print('\n')

##########################################################################

class clues():
    """
    Class definition for "Clues" object with attributes to specify whether
    object refers to "down" clues or "across" clues.  If down/accross attribute
    not specified, object returns both as a dict with "down/across" as keys.
    """
    def __init__(self):
        self.across = None
        self.down = None

    def parse_clues(self, clues_as_dict):
        self.across = clues_as_dict['across']
        self.down   = clues_as_dict['down']

    def __repr__(self):
        return pprint.pformat(self.__dict__)

##########################################################################

class answers():
    """
    Class definition for "Answers" object with attributes to specify whether
    object refers to "down" answers or "across" answers.  If down/accross
    attribute not specified, object returns both as a dict with "down/across" as
    keys.
    """
    def __init__(self):
        self.across = None
        self.down = None
        self.numbered = False

    def parse_answers(self, answers_as_dict):
        self.across = answers_as_dict['across']
        self.down   = answers_as_dict['down']

    def __repr__(self):
        return pprint.pformat(self.__dict__)

##########################################################################

class puzzle():
    """
    Class definition for "Puzzle" object.

    Attributes:
        - size (dimension of puzzle as rows and columns)
        - number of rows
        - number of columns
        - name of author
        - name of editor
        - date of puzzle
        - day of week
        - title (as applicable)
        - blank grid layout
        - solutions (with correct position of each letter in solution)
        - clues (see clues class)
        - answers (see answer class)

    Methods:
        - parse_puzzle: converts input dictionary defining puzzle into instance
                        of puzzle class by assigning relevant attributes
        - solution: outputs puzzle solution in proper grid format
        - blank: outputs empty puzzle grid with blocks in correct positions

    TO-DO:
        - clean up methods/attributes relating to answer numbering
    """

    def __init__(self):
        self.size = None
        self.rows = None
        self.cols = None
        self.author = None
        self.editor = None
        self.date = None
        self.dow = None
        self.title = None
        self.grid = None
        self.gridnums = None

        self.clues = clues()
        self.clues.across = None
        self.clues.down = None

        self.answers = answers()
        self.answers.numbered = False
        self.answers.across = None
        self.answers.down = None

    def parse_puzzle(self, puzzle_as_dict):
        self.size = puzzle_as_dict['size']
        self.author = puzzle_as_dict['author']
        self.editor = puzzle_as_dict['editor']
        self.date = puzzle_as_dict['date']
        self.dow = puzzle_as_dict['dow']
        self.title = puzzle_as_dict['title']
        self.grid = puzzle_as_dict['grid']
        self.gridnums = puzzle_as_dict['gridnums']

        self.rows = self.size['rows']
        self.cols = self.size['cols']

        self.clues.parse_clues(puzzle_as_dict['clues'])
        self.answers.parse_answers(puzzle_as_dict['answers'])

#         if self.answers.numbered == False:
#             number_answers(self)
#             self.answers.numbered = True

    def solution(self):
        print('\n**'+self.title+'**\n')
        grid = [sub.replace('.', u"\u25A0") for sub in self.grid]
        sol = []
        for row in range(self.rows):
            start = row * self.cols
            sol.append(grid[start:start+self.cols])
            print('[{0}]'.format(' '.join(map(str, sol[-1]))))

    def blank(self):
        print('\n**'+self.title+'**\n')
        grid = [char.replace('.', u"\u25A0") for char in self.grid]
        sol = []
        bl_list = []
        for row in range(self.rows):
            start = row * self.cols
            sol = grid[start:start+self.cols]
            for char in sol:
                bl_list.append(re.sub("[A-Z0-9]", "_", char))

        bl_list = [x[0] for x in bl_list]

        for row in range(self.rows):
            start = row * self.cols
            bl_ = bl_list[start:start+self.cols]
            print('[{0}]'.format('|'.join(map(str, bl_))))

    def __repr__(self):
        return pprint.pformat(self.__dict__)

##########################################################################
