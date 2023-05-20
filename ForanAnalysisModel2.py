import string
import tensorflow as tf
import string
import re
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from scipy.stats import rankdata
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
import tensorflow as tf


class Config:
    batch_size = 128
    validation_split = 0.15
    epochs = 5 # Number of Epochs to train
    model_path = "model.tf"
    output_dataset_path = "Output "
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    modes = ["training", "inference"]
    mode = modes[1]
    model_name = "distil_bert_base_en_uncased"
config = Config()

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
        # extracting symbols and characters
        tweet = re.sub(r'@\w+', "", tweet)
        tweet = re.sub(r'#\w+', "", tweet)
        tweet = re.sub(r'[~^0-9]', '',  tweet)
        tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        tweet.rstrip(string.punctuation)
        tweet = re.sub('[^A-Za-z\s]+', "", tweet)
        tweet = tweet.lower()
        tweet = tweet.split()
        # Lemmatization to normalise text
        tweet = [lemmatizer.lemmatize(word) for word in tweet if not word in STOPWORDS]
        tweet = ' '.join(tweet)
        filler = re.compile(fillerword_reg)
        tweet = filler.sub("", tweet)
        cleaned_data.append(tweet)
    return cleaned_data


def predict_sentiment2(text):
    # Load the sentiment analysis model (call load_model if needed)
    model = tf.keras.models.load_model('ML_Models/MultiLabelModel.h5', compile=False)
    
    # post = []
    # post.append(text)
    # postPD = pd.DataFrame(post)

    # Preprocess the text
    # CleanedPost = postPD.apply(lambda x:data_cleaning(x))
    # max_length=50

    # Perform sentiment analysis using the model
    # Example: Pass the preprocessed text through the model and get the sentiment prediction
    prediction = model.predict(text)
    #prediction = preprocessed_text

    return prediction
