# Gender Biasing Algorithm

A Python-based project for identifying possible gender bias in sentences by comparing gender-related subjects and their associated objects using natural language processing and similarity analysis.

## Overview

This project analyzes text to detect whether a sentence may contain gender bias. It uses:

- text preprocessing
- subject and object extraction
- gender-related word detection
- word similarity comparison using CBOW and Skip-Gram models

The goal is to estimate whether the language in a sentence leans toward a male or female association based on contextual similarity.

## Project structure

- `Gender_bias.py` – main bias detection logic
- `Data analysis code.py` – data analysis and experimentation scripts
- `practice.py` – additional exploratory code
- `text-and-id.txt` – dataset used for sentence analysis
- `pairs-label-training (1).txt` – label data used for evaluation
- `README.md` – project documentation

## Setup

### Requirements

- Python 3.9 or later
- pip

### Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

## Run the project

```bash
python "Gender_bias.py"
```

You can also run the analysis script:

```bash
python "Data analysis code.py"
```

## Important note

Some scripts contain hard-coded Windows file paths to local dataset files. Before running them, update those paths to match your machine or dataset location.

Example:

```python
file_name = 'D:\\Python\\Data sets\\...\\text-and-id.txt'
```

Replace this with the correct path for your environment.

## Method

The project follows a basic NLP workflow:

1. Clean the text and remove punctuation.
2. Extract sentence tokens.
3. Identify subject and object phrases.
4. Detect gender-related terms.
5. Compare similarity between gender-related terms and object terms.
6. Evaluate bias based on the difference in similarity scores.

## Data

The dataset contains sentence-level text and associated labels used for evaluating bias detection. The project also compares model outputs with training labels to estimate accuracy.

## Limitations

This is a research-oriented prototype and is not a production-grade fairness model. It depends heavily on:

- dataset quality
- language coverage
- word embeddings
- hard-coded preprocessing rules

## License

No license has been added to this repository yet.
