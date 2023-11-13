import glob
import glob
import json
import os
import random
from tqdm import tqdm
from image_to_poem.data.utils import extract_poem_info
from image_to_poem.utils import flatten_list

class KagglePoems:
    def __init__(self, poem_path = "data/kaggle_poems") -> None:
        if not poem_path.endswith("/"):
            poem_path += "/"
        self.paths = glob.glob(poem_path+"topics/*/*.txt")
        self.paths = [path.replace("\\", "/") for path in self.paths]
        self.poems = [self.read_poem(path) for path in tqdm(self.paths, desc="Reading poems")]
        self.words = [poem.split() for poem in self.poems]
        
    def read_poem(self, path):
        try:
            with open(path, 'r', encoding="utf8") as f:
                return f.read()
        except:
            print(f"Could not read {path}")
            return ""
        
    def get_example(self, example_index=None):
        example_index = example_index or random.randint(0, len(self.paths))
        example = self.paths[example_index]
        
        topic, title, author = extract_poem_info(example)
        print(f"File index: {example_index}")
        print(f"Topic: {topic}")
        print(f"Title: {title}")
        print(f"Author: {author}")
        print("-"*(8+max([len(topic), len(title), len(author)])))
        print(self.poems[example_index])
    
    @property    
    def vocab(self):
        return list(set(flatten_list(self.words)))
    
    @property
    def stats(self):
        return {
            "num_poems": len(self.poems),
            "num_words" : len(flatten_list(self.words)),
            "vocab_size": len(self.vocab),
            "avg_poem_length": sum([len(poem) for poem in self.words])/len(self.words),
        }

if __name__ == "__main__":
    poems = KagglePoems()
    poems.get_example()
    print(poems.stats)