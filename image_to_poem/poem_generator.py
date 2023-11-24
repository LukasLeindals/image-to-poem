from transformers import pipeline
from sentence_transformers import SentenceTransformer, util as st_utils
import numpy as np
from image_to_poem.language_model.gpt2 import GPT2Model

class PoemGenerator:
    def __init__(self, lm_model: str, n_candidates: int=10):
        print("Initializing image-to-text model...")
        self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        
        self.N = n_candidates
        
        # init GPT-2 language model
        print("Initializing language model...")
        self.lm_model = GPT2Model(pretrained_model=lm_model)

        
        
    def image_to_poem(self, image_paths):
        # make sure image_paths is a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        # create list for poems
        poems = [None]*len(image_paths)
        
        # get descriptions using image_to_text pipeline
        descriptions = self.image_to_text(image_paths)
        
        # generate poem for each description
        for i, desc in enumerate(descriptions):
            poems[i] = {}
            poems[i]["image_path"] = image_paths[i]
            poems[i]["image_desc"] = desc[0]["generated_text"]
            poems[i]["generated_poem"] = self.generate_poem(poems[i]["image_desc"])
            
        return poems
    
    def generate_poem(self, description: str):
        # print(description)
        
        poems = self.get_candidates(description)
        poem = self.select_poem(poems, description)
        
        return poem 
    
    def get_candidates(self, description: str):
        return self.lm_model.generate(prompt = description, num_return_sequences=self.N)
    
    def select_poem(self, candidates: list, description: str):
        # initialize BERT model
        bert = SentenceTransformer('all-MiniLM-L6-v2')
        
        # get poem embeddings and description embedding 
        poem_emb = bert.encode(candidates)
        desc_emb = bert.encode(description)
        
        # calculate similarity
        similarities = st_utils.cos_sim(poem_emb, desc_emb)
        similarities = np.array(similarities).flatten()
        
        # get the "best" = most similar candidate 
        best = np.argmax(similarities)
        return candidates[best]
        
if __name__ == "__main__":
    # initialize poem generator
    # poem_generator = PoemGenerator(lm_model = "models/language_models/model_20231123_215337/model/")
    poem_generator = PoemGenerator(lm_model = "models/language_models/model_20231123_190606/model/")
    
    # generate poems from images
    poems = poem_generator.image_to_poem(["data/poem_images/1.jpg", "data/poem_images/3.jpg"])
    print(poems)
