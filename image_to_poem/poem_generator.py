from transformers import pipeline
from sentence_transformers import SentenceTransformer, st_util
import numpy as np

class PoemGenerator:
    def __init__(self, n_candidates: int=10):
        print("Initializing image-to-text model...")
        self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        
        self.N = n_candidates
        
    def image_to_poem(self, image_paths):
        # make sure image_paths is a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        # create list for poems
        poems = [{}]*len(image_paths)
        
        # get descriptions using image_to_text pipeline
        descriptions = self.image_to_text(image_paths)
        
        # generate poem for each description
        for i, desc in enumerate(descriptions):
            poems[i]["image_path"] = image_paths[i]
            poems[i]["image_desc"] = desc[0]["generated_text"]
            poems[i]["generated_poem"] = self.generate_poem(poems[i]["image_desc"])
            
        return poems
    
    def generate_poem(self, description: str):
        # print(description)
        
        # poems = self.get_candidates()
        # poem = self.select_poem(poems)
        
        # return poem 
        return description
    
    def get_candidates():
        # call GPT 2 ? => make N candidates 
        pass
    
    def select_poem(candidates: list, description: str):
        # initialize BERT model
        bert = SentenceTransformer('all-MiniLM-L6-v2')
        
        # get poem embeddings and description embedding 
        poem_emb = bert.encode(candidates)
        desc_emb = bert.encode(description)
        
        # calculate similarity
        similarities = st_util.cos_sim(poem_emb, desc_emb)
        similarities = np.array(similarities).flatten()
        
        # get the "best" = most similar candidate 
        best = np.argmax(similarities)
        return candidates[best]
        
if __name__ == "__main__":
    # initialize poem generator
    poem_generator = PoemGenerator()
    
    # generate poems from images
    poems = poem_generator.image_to_poem(["data/poem_images/1.jpg", "data/poem_images/3.jpg"])
    print(poems)
