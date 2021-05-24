from nltk.tokenize import TweetTokenizer
import emojis
import emoji
import re

tt = TweetTokenizer()

def final_process(word):
    if emojis.count(word) > 0:
        return [word]
    elif len(re.sub(r'\W+', '', word)) > 1:
        return re.sub(r'\W+', ' ', word).split()
    else:
        return ''

def process_tweets(tweet):
    '''
    Takes in a single tweet (string of text, e.g. the 'tweet_text' key)
    and returns a processed, tokenized version of the tweet (list)
    '''
    # both emoji encoders exclude different emojis, so figured why not use both for increased coverage
    tweet = emoji.emojize(emojis.encode(tweet))
    # remove @ tags (usernames)
    tweet = re.sub('@[A-Za-z0-9_]+','', tweet)
    # remove commas from numbers (e.g. 1,000 -> 1000)
    tweet = re.sub('([0-9],[0-9])', lambda x: str(x.group(0)).replace(',',''), tweet)
    # remove strange unicode encodings and use NLTK TweetTokenizer to tokenize
    tweet = [word.encode('ascii', 'ignore').decode('ascii') if emojis.count(word) == 0 else word for word in tt.tokenize(tweet)]
    # remove nonalphanumeric words except for emojis
    tweet = [final_process(word) for word in tweet if final_process(word) != '']
    return [item for sublist in tweet for item in sublist]
