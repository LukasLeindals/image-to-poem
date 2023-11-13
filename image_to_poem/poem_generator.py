from transformers import pipeline

class PoemGenerator:
    def __init__(self):
        print("Initializing image-to-text model...")
        self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        
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
        return description
    
    def get_candidates():
        pass
    
    def select_poem():
        pass
        
if __name__ == "__main__":
    # initialize poem generator
    poem_generator = PoemGenerator()
    
    # generate poems from images
    poems = poem_generator.image_to_poem(["data/poem_images/1.jpg", "data/poem_images/3.jpg"])
    print(poems)
