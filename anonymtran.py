
!pip install transformers datasets seqeval torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load model and tokenizer
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Load NER pipeline-t
ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def load_data(file_path):
    with open(file_path, 'r') as f:
        TRAIN_DATA = json.load(f)["examples"]
    return TRAIN_DATA

from collections import defaultdict

texts = [text for text, ann in TRAIN_DATA]
gold_labels = [ann["entities"] for _, ann in TRAIN_DATA]

def predict_entities(text):
    return [(ent['start'], ent['end'], ent['entity_group']) for ent in ner_pipe(text)]

predictions = [predict_entities(text) for text in texts]

print(predictions)

print(gold_labels)

from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

# Label normalization map
LABEL_MAP = {
    'PER': 'PERSON', 'PERSON': 'PERSON',
    'LOC': 'GPE', 'GPE': 'GPE',
    'ORG': 'ORG',
    'MISC': 'MISC', 'FAC': 'MISC', 'EVENT': 'MISC', 'PRODUCT': 'MISC',
    'DATE': 'MISC', 'PERCENT': 'MISC', 'LANGUAGE': 'MISC',
}

def normalize_label(label):
    return LABEL_MAP.get(label, label)

def normalize_entities(entities):
    return [(start, end, normalize_label(label)) for (start, end, label) in entities]

def iou(span1, span2):
    # Calculate Intersection over Union for span similarity
    start1, end1 = span1
    start2, end2 = span2
    inter = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return inter / union if union > 0 else 0

def entity_match(pred_ent, true_ent, iou_threshold=0.5):
    (p_start, p_end, p_label) = pred_ent
    for idx, (t_start, t_end, t_label) in enumerate(true_ent):
        if normalize_label(p_label) == normalize_label(t_label):
            if iou((p_start, p_end), (t_start, t_end)) >= iou_threshold:
                return idx
    return -1

def evaluate_ner(predictions, truths):
    pred_total, true_total, matched = 0, 0, 0

    for pred_sample, true_sample in zip(predictions, truths):
        pred_entities = normalize_entities(pred_sample)
        true_entities = normalize_entities(true_sample)
        true_used = set()
        pred_total += len(pred_entities)
        true_total += len(true_entities)

        for pred_ent in pred_entities:
            match_idx = entity_match(pred_ent, true_entities)
            if match_idx != -1 and match_idx not in true_used:
                matched += 1
                true_used.add(match_idx)

    precision = matched / pred_total if pred_total else 0
    recall = matched / true_total if true_total else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'Matched': matched,
        'Predicted': pred_total,
        'Ground Truth': true_total
    }
results = evaluate_ner(predictions, gold_labels)
print(results)


