from transformers import pipeline
import numpy as np
from image_to_poem.language_model.gpt2 import GPT2Model
from image_to_poem.similarity.similarity_scoring import BertSimilarityModel

class PoemGenerator:
    def __init__(self, lm_model: str, sim_model: str, n_candidates: int=10):
        self.N = n_candidates   # #poem candidates to generate 
        
        print("Initializing image-to-text model...")
        self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        
        # init GPT-2 language model
        print("Initializing language model...")
        self.lm_model = GPT2Model(pretrained_model=lm_model)
        
        # init BERT similarity model
        print("Initializing BERT similarity model...")
        self.sim_model = BertSimilarityModel.from_model_dir(sim_model)

    def image_to_poem(self, image_paths, **kwargs):
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
            poems[i]["generated_poem"] = self.generate_poem(poems[i]["image_desc"], **kwargs)
            
        return poems
    
    def generate_poem(self, description: str, **kwargs):
        # generate N poems 
        poems = self.get_candidates(description, **kwargs)
        # select "best match"
        poem = self.select_poem(poems, description)
        # return final poem 
        return poem 
    
    def get_candidates(self, description: str, **kwargs):
        # use LM to generate N poems from the description 
        return self.lm_model.generate(prompt = description, num_return_sequences=self.N, **kwargs)
    
    def select_poem(self, candidates: list, description: str):
        # collect similarity score for each poem candidate 
        sim_scores = [0]*len(candidates)
        for i,poem in enumerate(candidates):
            sim_scores[i] = self.sim_model.similarity(description, poem)
        # find poem with highest score 
        best = np.argmax(sim_scores)
        return candidates[best]
        
if __name__ == "__main__":
    # initialize poem generator
    poem_generator = PoemGenerator(lm_model="models/language_models/max_len-500", 
                                   sim_model="models/similarity/model_20231129_221129",
                                   n_candidates=5)
    
    # generate poems from images
    poems = poem_generator.image_to_poem(["data/poem_images/1.jpg", "data/poem_images/3.jpg"])
    print(poems)
