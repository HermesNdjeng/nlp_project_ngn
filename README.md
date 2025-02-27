# Detecting AI-Generated Research Abstracts with SciBERT and spaCy

This project fine-tunes SciBERT within a spaCy-based pipeline to detect automatically generated research abstracts. It combines Transformer embeddings with a text classification ensemble, leveraging both shallow statistical features and deep contextual signals to distinguish human-written from machine-generated text.

## Project Structure

```
├── config.cfg            # spaCy pipeline configuration file
├── nlp_project_ngn.ipynb        # Annotated Jupyter Notebook with full workflow
├── README.md             # Project documentation
```

## Dataset

We use the FullyGenerated set from Liyanage et al. (2022). The dataset consists of fully generated and human-written abstracts.
- 80% training / 20% testing split (stratified).
- Converted into spaCy DocBin format for efficient processing (train.spacy, test.spacy).

## Installation & Setup

### Install Dependencies

```bash
pip install spacy-transformers
```

### Download the Dataset

```bash
wget https://github.com/vijini/GeneratedTextDetection/archive/refs/heads/main.zip
unzip main
```

### Train the Model

```bash
python -m spacy train config.cfg --output ./output --gpu-id 0
```

### Evaluate the Model

```bash
python -m spacy evaluate ./output/model-best test.spacy
```


## Model Architecture

- **Transformer Component**: SciBERT (allenai/scibert_scivocab_uncased) generates contextual embeddings.
- **Text Classification**: TextCatEnsemble.v2 with:
    - **Bag-of-Words (BoW)**: Captures lexical frequency patterns.
    - **Transformer Listener**: Leverages SciBERT embeddings for deeper semantic analysis.

## Results & Performance

- High accuracy in distinguishing human-written vs. machine-generated abstracts.
- Misclassifications occur when AI-generated text is highly fluent and contextually accurate.
- Performance bottlenecks due to limited training data and overfitting risks.

## Model Deployment

The trained model is packaged for distribution:

### Package the Model

```bash
python -m spacy package ./output/model-best ./my_model_package --build wheel --create-meta
```

### Install on a Remote Machine

```bash
pip install https://github.com/yourusername/yourrepo/releases/download/v1.0.0/your_model.whl
```

Alternatively, install from the .tar.gz model file:

```bash
pip install en_model_textcat_ngn-0.1.0.tar.gz
```

## Contributors

This project was developed by:
- **Harold** 
- **Aymrick**
- **Hermers**

## Citation

If you use this work, please cite:

```bibtex
@article{liyanage2022generatedtextdetection,
    title={Detecting AI-Generated Scientific Text},
    author={Liyanage, Vijini and et al.},
    journal={ArXiv},
    year={2022}
}
```