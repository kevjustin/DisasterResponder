import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import spacy
import glob
import re
from sklearn.utils import resample
import streamlit as st
import random

def load_data_from_directory(directory_path, sample_fraction=0.1):
    all_files = glob.glob(f"{directory_path}/*.tsv")
    data_list = []
    
    for file_path in all_files:
        data = pd.read_csv(file_path, sep='\t', names=['tweet_id', 'tweet_text', 'label'], header=None)
        data_list.append(data)
    
    combined_data = pd.concat(data_list, ignore_index=True)
    
    # `sample_fraction=0.3` for 30% of the data
    if sample_fraction < 0.3:
        combined_data = combined_data.sample(frac=sample_fraction, random_state=42)
    
    return combined_data

def preprocess_and_balance_data(data):
    data['label'] = data['label'].astype('category').cat.codes
    
    def clean_text(text):
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters, numbers, and punctuations
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    data['tweet_text'] = data['tweet_text'].apply(clean_text)
    
    max_count = data['label'].value_counts().max()

    balanced_data = []
    for label in data['label'].unique():
        class_data = data[data['label'] == label]
        balanced_class_data = resample(class_data,
                                       replace=True,
                                       n_samples=max_count,
                                       random_state=42)
        balanced_data.append(balanced_class_data)

    balanced_data = pd.concat(balanced_data, ignore_index=True)
    
    return balanced_data

def extract_locations(texts):
    locations = []
    for text in texts:
        doc = nlp(text)
        locs = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        locations.append(locs)
    return locations

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    report = classification_report(labels, preds, zero_division=0, output_dict=True)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }

def classify_tweet(tweet_text, model, tokenizer, nlp, label_mapping):

    inputs = tokenizer(
        tweet_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = (torch.argmax(logits, dim=-1).item())//2

    doc = nlp(tweet_text)
    extracted_locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    predicted_label_name = label_mapping.get(predicted_label, "Unknown")

    return predicted_label_name, extracted_locations

data = load_data_from_directory('CrisisNLP_labeled_data_crowdflower')
data = preprocess_and_balance_data(data)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['tweet_text'].values,
    data['label'].values,
    test_size=0.2,
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CrisisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = CrisisDataset(train_texts, train_labels, tokenizer)
test_dataset = CrisisDataset(test_texts, test_labels, tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Load or train the model
model_path = './saved_model'
if os.path.exists(model_path):
    print("Loading saved model...")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
else:
    print("Training new model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['label'].unique()))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Location Extraction with SpaCy
nlp = spacy.load("en_core_web_sm")
test_locations = extract_locations(test_texts)

trainer.compute_metrics = compute_metrics
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
test_metrics = compute_metrics(predictions)

# Streamlit User Interface
st.title("Tweet Classifier and Location Extractor")
st.subheader("Evaluation Metrics")
st.json(test_metrics)

random.seed()
random_indices = random.sample(range(len(test_texts)), 10)
st.subheader("Predictions and True Labels (10 Examples)")
for i in random_indices:
    st.write(f"**Tweet**: {test_texts[i]}")
    st.write(f"**Predicted Label**: {predicted_labels[i]} | **True Label**: {test_labels[i]}")
    st.write("---")

st.subheader("Extracted Locations (10 Examples)")
for i in random_indices:
    st.write(f"**Tweet**: {test_texts[i]}")
    st.write(f"**Extracted Locations**: {', '.join(test_locations[i]) if test_locations[i] else 'None'}")
    st.write("---")

label_mapping = {idx: category for idx, category in enumerate(data['label'].astype('category').cat.categories)}

# Mock Tweet Classification
st.header("Mock Tweet Classifier")
user_tweet = st.text_area("Enter a tweet to classify:")
if st.button("Classify"):
    if user_tweet.strip():
        predicted_label, extracted_locations = classify_tweet(user_tweet, model, tokenizer, nlp, label_mapping)
        st.subheader("Classification Result:")
        st.write(f"Predicted Label: {predicted_label}")
        st.subheader("Extracted Locations:")
        st.write(", ".join(extracted_locations) if extracted_locations else "None")
    else:
        st.warning("Please enter a valid tweet.")

# To run: streamlit run DisasterResponder.py