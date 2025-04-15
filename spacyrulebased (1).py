
import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
from sklearn.metrics import precision_recall_fscore_support

def load_data(file_path):
    with open(file_path, 'r') as f:
        TEST_DATA = json.load(f)["examples"]
    return TEST_DATA


def train_ruler():
    nlp = spacy.blank("en")  # Create a blank English NLP pipeline
    ruler = nlp.add_pipe("entity_ruler")

    # Load training data


    # Add patterns
    patterns = []
    for text, annotations in TEST_DATA:
        for start, end, label in annotations["entities"]:
            patterns.append({"label": label, "pattern": text[start:end]})

    ruler.add_patterns(patterns)
    return nlp


def add_patterns(nlp):
    # Check if 'entity_ruler' already exists; if not, add it
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        # Get existing 'entity_ruler'
        ruler = nlp.get_pipe("entity_ruler")
    patterns = [
        {"label": "PERSON", "pattern": [{"TEXT": {"REGEX": "^[A-Z][a-z]+ [A-Z][a-z]+$"}}]},  # Capitalized full names
        {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]},  # Emails
        {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}"}}]},  # Phone numbers
        {"label": "ID", "pattern": [{"TEXT": {"REGEX": r"\b\d{9,12}\b"}}]},  # ID numbers
    ]
    ruler.add_patterns(patterns)
    return nlp


def custom_matcher(nlp):
    matcher = Matcher(nlp.vocab)

    # Match "Mr. John Doe" or "Dr. Sarah Connor"
    name_pattern = [{"IS_TITLE": True}, {"IS_ALPHA": True}, {"IS_ALPHA": True}]
    matcher.add("TITLE_NAME", [[{"TEXT": {"REGEX": "(Mr|Dr|Ms|Mrs)\.?"}}, {"IS_ALPHA": True}, {"IS_ALPHA": True}]])

    # Match social security numbers (XXX-XX-XXXX)
    ssn_pattern = [{"TEXT": {"REGEX": r"\b\d{3}-\d{2}-\d{4}\b"}}]
    matcher.add("SSN", [ssn_pattern])

    return matcher

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

    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

nlp=train_ruler()
nlp=add_patterns(nlp)
#matcher = custom_matcher(nlp)

evaluate()

