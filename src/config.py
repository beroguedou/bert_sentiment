import transformers


MAX_LEN = 512
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 3
EPOCHS = 10
BERT_PATH = "../inputs/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE= "../inputs/IMDB_Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)