import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Download necessary resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)

# Load the file in chunks
reader = pd.read_json('yelp_academic_dataset_tip.json', lines=True, chunksize=100)

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


# STEP 1 : Training the model : 

from gensim.models import FastText

#Making sure that the tokens are coverted to sentences with len of 10
tokenized = [all_processed_tokens[i:i+10] for i in range(0, len(all_processed_tokens), 10)]

model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=2, epochs=10,bucket=100000)

model.save("my_fasttext_model.model")
model = FastText.load("my_fasttext_model.model")

word = "food"

print(f"10 similar words to '{word}':")
for w, score in model.wv.most_similar(positive=[word], topn=10):
    print(f"{w} ({score:.2f})")

print(f"\n10 opposite words to '{word}':")
for w, score in model.wv.most_similar(negative=[word], topn=10):
    print(f"{w} ({score:.2f})")


# STEP 2 Pretrained FastText Model 
from gensim.models import FastText
import gensim.downloader as api

print("Loading pretrained word vectors... please wait!")
pretrained = api.load("glove-wiki-gigaword-100")

# Build FastText model using pretrained vectors
word_list = list(pretrained.key_to_index.keys())[:50000]
tokenized2 = [word_list[i:i+10] for i in range(0, len(word_list), 10)]

model2 = FastText(sentences=tokenized2, vector_size=100, window=5, min_count=1, epochs=5,bucket=100000)

test_words = ["food", "service", "place"]

for word in test_words:
    print("="*45)
    print(f"  RESULTS FOR INPUT WORD: '{word}'")
    print("="*45)

    print(f"\n[+] 10 Similar words to '{word}':")
    similar = model2.wv.most_similar(positive=[word], topn=10)
    for i, (w, score) in enumerate(similar, 1):
        print(f" {i}. {w:<15} | Score: {score:.4f}")

    print(f"\n[-] 10 Opposite words to '{word}':")
    opposites = model2.wv.most_similar(negative=[word], topn=10)
    for i, (w, score) in enumerate(opposites, 1):
        print(f" {i}. {w:<15} | Score: {score:.4f}")

    print("\n" + "-"*45)


# STEP 3  Pretrained with Yelp 

print("Updating model with Yelp data...")
model2.build_vocab(tokenized, update=True)
model2.train(tokenized, total_examples=len(tokenized), epochs=10)

print("Model updated successfully!\n")

for word in test_words:
    print("="*45)
    print(f"  RESULTS FOR INPUT WORD: '{word}'")
    print("="*45)

    print(f"\n[+] 10 Similar words to '{word}':")
    similar = model2.wv.most_similar(positive=[word], topn=10)
    for i, (w, score) in enumerate(similar, 1):
        print(f" {i}. {w:<15} | Score: {score:.4f}")

    print(f"\n[-] 10 Opposite words to '{word}':")
    opposites = model2.wv.most_similar(negative=[word], topn=10)
    for i, (w, score) in enumerate(opposites, 1):
        print(f" {i}. {w:<15} | Score: {score:.4f}")

    print("\n" + "-"*45)
