import pandas as pd
import config
from text_preprocessing import clean_text, text_embedding
from nltk import word_tokenize

if __name__ == "__main__":
    embedding_dict = text_embedding.load_vectors(config.EMEDDING_VEC_FILE)
    for fname in [config.DEV_TRAIN_FILE, config.DEV_TEST_FILE]:
        dataframe = pd.read_csv(fname)
        dataframe["text"] = dataframe["text"].apply(clean_text.clean())
        
        dataframe.to_csv(fname, index=False)
        #dataframe["text"] = dataframe["text"].apply(clean_text.snowball_stemmer)
        #dataframe["text"] = dataframe["text"].apply(clean_text.lemmatize_word)

        vectors = []
        for line in dataframe["text"].values:
            vectors.append(
                text_embedding.sentence_embedding(
                    sentence = line,
                    embedding_dict = embedding_dict,
                    tokenizer = word_tokenize,
                    dim = config.EMBEDDING_DIM
                )
            )

        new_dataframe = pd.DataFrame(vectors)
        if "train" in fname:
            new_dataframe["kfold"] = dataframe["kfold"].values
            new_dataframe["target"] = dataframe["target"].values
        new_dataframe.to_csv(fname, index=False)