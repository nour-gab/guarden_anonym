# Guarden Anonymization

This project evaluates multiple approaches for Named Entity Recognition (NER) in the context of anonymizing sensitive academic and recommendation texts.
To build and compare different anonymization techniques for detecting named entities like names, organizations, locations, and other personally identifiable information (PII) using:

- 🔹 Hugging Face Transformer Models
- 🔸 spaCy Rule-based Patterns
- 🔹 spaCy with `en_core_web_trf` transformer model


## 🛠️ Models Used

### Hugging Face Transformers
- **Model 1:** `dslim/bert-base-NER`
- **Model 2:** `Jean-Baptiste/roberta-large-ner-english`

### spaCy Rule-based Anonymizer
- Using a pattern-matching system with spaCy's `Matcher` and `EntityRuler`.

###  spaCy Transformer Model
- Using `en_core_web_trf` — spaCy’s transformer pipeline trained on OntoNotes 5.


##  Evaluation Metrics

Each model was evaluated using a custom-labeled dataset with the following metrics:

| Method                            | Precision | Recall | F1 Score |
|----------------------------------|-----------|--------|----------|
| `dslim/bert-base-NER`            |  **30.51%**   | **46.75%** | **36.92%**  |
| `Jean-Baptiste/roberta-large-ner-english`  |  **8%**   | **10%** | **8%**  |
| Rule-based spaCy Anonymizer      |  **90.3%**   | **87.26%** | **88%**  |
| spaCy `en_core_web_trf`          |  **100%**   | **98.8%** | **99.4%**  |
