from gensim.models import Doc2Vec
class CopypastaDetector:
    def __init__(self, model_path: str, threshold: float = 0.8):
        self.threshold = threshold
        self.model = Doc2Vec.load(model_path)

    def detect_copypasta(self, new_text: str) -> bool:
        new_vec = self.model.infer_vector(new_text.split())
        sims = self.model.docvecs.most_similar([new_vec], topn=len(self.model.docvecs))
        for sim in sims:
            if sim[1] > self.threshold:
                return True
        return False
    
def check_copypasta(text:str, threshold:float=0.85) -> bool:
    model_path = "doc2vec_model"
    detector = CopypastaDetector(model_path, threshold)
    return detector.detect_copypasta(text)


text_to_classify = "did you know, that"
is_copypasta = check_copypasta(text_to_classify)
print(is_copypasta) # True