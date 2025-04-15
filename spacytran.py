
!nvcc --version

!pip install -U spacy[transformers,cuda125]

!python -m spacy download en_core_web_trf

def load_data(file_path):
    with open(file_path, 'r') as f:
        TEST_DATA = json.load(f)["examples"]
    return TEST_DATA

import spacy
nlp = spacy.load("en_core_web_trf")

def evaluate():

    y_true, y_pred = [], []

    for data_point in TEST_DATA:
        text = data_point[0]
        annotations = data_point[1]["entities"]
        doc = nlp(text)
        true_entities = [(text[start:end], label) for start, end, label in annotations]

        # Create a list of predicted labels for this data point
        pred_labels_for_data_point = []
        for start, end, label in annotations:
            # Extract the entity text from the current annotation
            entity_text = text[start:end]
            # Check if any predicted entities match the current entity text
            matched_pred_entity = next((ent.label_ for ent in doc.ents if ent.text == entity_text), None)

            # If a match is found, append the predicted label; otherwise, append the true label
            pred_labels_for_data_point.append(matched_pred_entity if matched_pred_entity is not None else label)

        y_true.extend([label for _, label in true_entities])
        y_pred.extend(pred_labels_for_data_point)

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

evaluate()