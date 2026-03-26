import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Download necessary resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)

# Load the file in chunks
reader = pd.read_json('yelp_academic_dataset_tip.json', lines=True, chunksize=10000)

all_processed_tokens = []

for chunk in reader:
    # 1. Lowercase and Split (Tokenize)
    # .str allows us to apply string functions to the whole column
    clean_text = chunk['text'].str.lower()
    
    for text in clean_text:
        # 2. Remove Punctuation
        text_no_punct = text.translate(punctuation_table)
        
        # 3. Tokenize and Remove Stopwords
        words = text_no_punct.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        all_processed_tokens.extend(filtered_words)

    # Optional: break after first chunk for testing so you don't wait forever
    # break 

print(all_processed_tokens[:100]) # Print first 20 tokens