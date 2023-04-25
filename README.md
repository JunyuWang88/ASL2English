# ASL2English

## Get started (Optional)

1. Run preprocess.py to generate asl training data from tree structure of english gloss data
2. Run train.py to train the model and save to disk

## Train

python -m spacy init fill-config base_config.cfg config.cfg  
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy

## Load and Use

after_training.py

