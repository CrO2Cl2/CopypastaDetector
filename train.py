from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Prepare the dataset
def prepare_dataset(filepath:str)->list[TaggedDocument]:
    with open(filepath, encoding="utf-8") as f:
        data = [line.strip() for line in f]
    return [TaggedDocument(doc.split(), [str(i)]) for i, doc in enumerate(data)]

# Training function
def train(train_filepath:str, output_path:str)->None:
    train_data = prepare_dataset(train_filepath)
    model = Doc2Vec(vector_size=100, min_count=2, epochs=40)
    model.build_vocab(train_data)
    model.corpus_count = len(train_data)  # set the corpus_count attribute
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(output_path)


# calling the function
train("copypasta_dataset.txt","doc2vec_model")
