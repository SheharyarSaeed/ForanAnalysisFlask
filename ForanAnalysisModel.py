import pandas as pd
import re
import string
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()  
from keras.preprocessing.text import tokenizer_from_json
import joblib
import tensorflow as tf
from keras.utils import pad_sequences

def data_cleaning(data):
    cleaned_data = []
    fillerWord = (
    "so", "yeah", "okay", "um", "uh", "mmm", "ahan", "uh", "huh", "ahm", "oh", "sooo", "uh", "huh", "yeh", "yah", "hmm",
    "bye")
    fillerword_reg = "bye[.,]|so[.,]|yeah[.,]|okay[.,]|um[.,]|uh[.,]|mmm[.,]|ahan[.,]|uh[.,]|huh[.,]|ahm[.,]|oh[.,]|sooo[.,]|uh[.,]|huh[.,]|yeh[.,]|yah[.,]|hmm[.,]"
    STOPWORDS = set(stopwords.words('english'))
    remove = ["doesn't", "not", "nor", "neither", "isn't", "hadn't", "mightn't", "needn't", "wasn't"]
    for i in remove:
        STOPWORDS.discard(i)

    STOPWORDS.add(fillerWord)
    for i in range(len(data)):
        tweet = re.sub("#", "", data[i])  # extracting hashtags
        tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)  # extracting links
        html = re.compile(r'<.*?>')  # extracting html tags
        tweet = html.sub(r"", tweet)
        # extracting symbols and characters
        tweet = re.sub(r'@\w+', "", tweet)
        tweet = re.sub(r'#\w+', "", tweet)
        tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        tweet.rstrip(string.punctuation)
        tweet = re.sub('[^A-Za-z\s]+', "", tweet)
        tweet = re.sub(r'[~^0-9]', '',  tweet)
        tweet = tweet.lower()
        tweet = tweet.split()
        # Lemmatization to normalise text
        tweet = [lemmatizer.lemmatize(word) for word in tweet if not word in STOPWORDS]
        tweet = ' '.join(tweet)
        filler = re.compile(fillerword_reg)
        tweet = filler.sub("", tweet)
        cleaned_data.append(tweet)
    return cleaned_data

def GetCleanedText(text):
    # post = []
    # post.append(text)
    postPD = pd.DataFrame(text)

    # Preprocess the text
    CleanedPost = postPD.apply(lambda x:data_cleaning(x))
    return CleanedPost

def predict_sentiment(text):
    # Load the sentiment analysis model (call load_model if needed)
    model = tf.keras.models.load_model('ML_Models/FYP_Model_New1.h5', compile=False)
    TransPkl = joblib.load('ML_Models/tokenizer_json_New.pkl')
    tokenizer = tokenizer_from_json(TransPkl)
    #modelWeight = tf.keras.models.load_model('joblibModels/model1Weights.h5')

    #model.set_weights(modelWeight)

    # Preprocess the text
    max_length=50

    # print(text)
    postsequence = text.apply(lambda x:tokenizer.texts_to_sequences(x))
    # print(postsequence)
    postpadding = pad_sequences(postsequence.loc[:,0].tolist(),maxlen=max_length,padding='post')
    # print(postpadding)

    # Perform sentiment analysis using the model
    # Example: Pass the preprocessed text through the model and get the sentiment prediction
    prediction = model.predict(postpadding)
    #prediction = preprocessed_text

    return prediction

def getScrapperData(input):
    import requests
    import csv
    import time

    subreddit = input # Replace with the desired subreddit
    post_limit = 10

    headers = {
        'client_id':'xjT8d3zBYks67rpmDswShQ',
                        'client_secret':'NKJVFGaEz3EgLAL80B8fv3HJ1cJfQQ',
                        'user_agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42',
                        'username':'Kakashi_987',
                        'password':'Shazam@123'
    }

    url = f'https://www.reddit.com/r/{subreddit}/new.json?limit={post_limit}'

    max_retries = 10
    retry_delay = 1  # seconds

    # Function to handle the backoff and retry mechanism
    def handle_rate_limit(response, retries):
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 0))
            wait_time = 2 ** retries * retry_delay  # Exponential backoff
            print(f"Rate limit exceeded. Retrying after {wait_time} seconds...")
            time.sleep(wait_time)
            return True
        return False

    # Make the request with retry logic
    def make_request_with_retry(url, headers):
        retries = 0
        while retries < max_retries:
            response = requests.get(url, headers=headers)
            if not handle_rate_limit(response, retries):
                return response
            retries += 1
        return None

    response = make_request_with_retry(url, headers)

    if response is not None and response.status_code == 200:
        data = response.json()
        
        # Extract post data from the JSON response
        posts = data['data']['children']
        
        # Specify the CSV file path to save the data
        #csv_file_path = 'posts.csv'
        post_data = []
        
        # # Open the CSV file in write mode
        # with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        #     writer = csv.writer(csv_file)
            
        #     # Write the header row
        #     writer.writerow(['Title', 'Author', 'Created UTC', 'URL', 'Self Text'])

            
        #     # Write each post's data as a row in the CSV file
        #     for post in posts:
        #         title = post['data']['title']
        #         author = post['data']['author']
        #         created = post['data']['created_utc']
        #         url = post['data']['url']
        #         self_text = post['data']['selftext']
                
        #         writer.writerow([title, author, created, url, self_text])
        #print(f"Posts saved to {csv_file_path}")

        for post in posts:
            title = post['data']['title']
            #author = post['data']['author']
            #created = post['data']['created_utc']
            url = post['data']['url']
            self_text = post['data']['selftext']
            
            # Append post data to the list
            # post_data.append([title, author, created, url, self_text])
            post_data.append([title, url, self_text])

        # Create a DataFrame from the post data
        #df = pd.DataFrame(post_data, columns=['Title', 'Author', 'Created UTC', 'URL', 'Self Text'])
        df = pd.DataFrame(post_data, columns=['Title', 'URL', 'SelfText'])
        print("Data fetch successfull!")
        return df

    elif response is not None:
        print(f"Request failed with status code {response.status_code}")
    else:
        print("Request failed. Maximum number of retries reached.")


