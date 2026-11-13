# Disaster Responder

Disaster Responder is a machine learning project that classifies disaster-related tweets and extracts place names mentioned in them. It fine-tunes BERT on labeled CrisisNLP tweets and provides a Streamlit interface for exploring evaluation results and trying custom tweets.

## Features

- **Tweet classification** — Predicts one CrisisNLP information category per tweet.
- **Place-name extraction** — Uses spaCy to identify geopolitical entities such as cities and countries.
- **Evaluation dashboard** — Displays accuracy, weighted precision, weighted recall, and weighted F1, alongside 10 randomly selected evaluation examples.
- **Custom tweet demo** — Lets you paste a tweet and see a predicted label and any detected place names.

## Setup

From the repository root, install the Python dependencies and the spaCy English model.

```bash
python -m pip install pandas numpy scikit-learn torch "transformers>=4.46,<5" accelerate spacy streamlit
python -m spacy download en_core_web_sm
```

Place the CrisisNLP CrowdFlower TSV files in a folder named `CrisisNLP_labeled_data_crowdflower` next to `DisasterResponder.py`. The loader expects three tab-separated columns in this order: tweet ID, tweet text, and category label. It reads every `.tsv` file in the folder with no header and samples 10% of the combined rows by default.

## Run locally

```bash
streamlit run DisasterResponder.py
```

On the first run, the script downloads `bert-base-uncased`, fine-tunes it for three epochs, and saves the model and tokenizer to `saved_model/`. Later runs load that saved model. The dataset folder is required on each run because the dashboard rebuilds the evaluation split.

## How it works

1. Load and sample the labeled tweets, then remove URLs, mentions, hashtags, and nonalphabetic characters and lowercase the text.
2. Balance the categories by resampling, then split the resulting rows into 80% training and 20% evaluation.
3. Fine-tune or load BERT and evaluate its predictions on the evaluation split.
4. Run spaCy named-entity recognition and display the extracted place names in Streamlit.

## Current limitations

- Resampling happens before the train/evaluation split, so copies of the same tweet can appear in both sets and inflate the displayed metrics. The evaluation split also serves as the validation set during training.
- The custom tweet demo divides the predicted class index by two and builds its label lookup from numeric codes. As a result, the displayed class ID can differ from BERT's direct prediction, and the original CrisisNLP category names are unavailable in the interface.
- Extracted locations are place names mentioned in the text. The app does not resolve them to coordinates or verify where an incident occurred.
