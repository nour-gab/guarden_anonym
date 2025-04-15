
!pip install seqeval

from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report
import re


def load_data(file_path):
    with open(file_path, 'r') as f:
        TEST_DATA = json.load(f)["examples"]
    return TEST_DATA

ner_pipeline = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="simple")

def get_predicted_entities(text):
    predictions = ner_pipeline(text)
    spans = set()
    for ent in predictions:
        entity_text = text[ent['start']:ent['end']]
        label = ent['entity_group']
        spans.add((ent['start'], ent['end'], label))
    return spans

def convert_to_seqeval_format(text, spans):
    tokens = text.split()
    tags = ['O'] * len(tokens)
    for start, end, label in spans:
        # Find which tokens this span overlaps with
        char_idx = 0
        for i, token in enumerate(tokens):
            token_start = text.find(token, char_idx)
            token_end = token_start + len(token)
            char_idx = token_end
            if start < token_end and end > token_start:
                if tags[i] == 'O':
                    tags[i] = f"B-{label}" if start == token_start else f"I-{label}"
    return tags

def evaluate_ner(test_data):
    y_true_all = []
    y_pred_all = []

    for text, annotation in test_data:
        true_spans = annotation['entities']
        pred_spans = get_predicted_entities(text)

        true_tags = convert_to_seqeval_format(text, true_spans)
        pred_tags = convert_to_seqeval_format(text, pred_spans)

        y_true_all.append(true_tags)
        y_pred_all.append(pred_tags)

        # Uncomment to see detailed side-by-side comparisons
        print(f"\nText: {text}")
        print(f"True Tags: {true_tags}")
        print(f"Pred Tags: {pred_tags}")

    print("\nSeqEval Classification Report:")
    print(classification_report(y_true_all, y_pred_all))

evaluate_ner(TEST_DATA)

