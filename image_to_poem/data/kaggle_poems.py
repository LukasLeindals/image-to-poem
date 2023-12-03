import glob
import random
from tqdm import tqdm
from image_to_poem.utils import flatten_list

import re

def extract_poem_info(path):
    """
    Extracts the poem info from the path of a poem file
    
    Args:
        path (str): The path to the poem file
    
    Returns:
        (str, str, str): The category, title, and author of the poem
    """
    # ensure path is in unix format
    path = path.replace("\\", "/")
    
    # get category (topic/forms)
    cat = path.split('/')[-2]
    
    # get title and author
    split_on_capitals = lambda x: re.sub(r"(\w)([A-Z])", r"\1 \2", x)
    name = path.split('/')[-1].replace('.txt', '')
    title, author = name.split('Poemby')
    title, author = split_on_capitals(title), split_on_capitals(author)
    
    # remove first two words from title as these describe the category
    title = ' '.join(title.split(' ')[2:])
    
    return cat, title, author

class KagglePoems:
    """
    Class for handling the Kaggle poems dataset.
    """
    def __init__(self, poem_path = "data/kaggle_poems/topics/", poem_ext = "txt", max_poems = None) -> None:
        """
        Initializes the KagglePoems class.
        
        Parameters
        ----------
        poem_path (str, optional):
            Path to the poems. Defaults to "data/kaggle_poems/topics/", meaning the topics subset will not be included.
        poem_ext (str, optional):
            File extension of the poems. Defaults to "txt".
        max_poems (int, optional):
            Maximum number of poems to include. Defaults to None, meaning all poems will be included.
        """
        self.poem_path = poem_path
        self.poem_ext = poem_ext
        self.max_poems = max_poems
        
        # get paths
        self.get_poem_paths(poem_path)
        
        # filter out unnecesary poems
        if max_poems is not None:
            self.paths = self.paths[:min(max_poems, len(self.paths))]
        
        # read poems
        self.poems = [self.read_poem(path) for path in tqdm(self.paths, desc="Reading poems")]
        
        # split poems into words
        self.words = [poem.split() for poem in self.poems]
        
    def get_poem_paths(self, poem_path):
        """
        Get file paths for all poems.
        
        Parameters
        ----------
        poem_path (str):
            Path to the poems folder.
        """
        # add / to path
        if not poem_path.endswith("/"):
            poem_path += "/"
            
        # allow all files to be found recursively
        poem_path += "**"
        
        # add extension filter
        if self.poem_ext is not None:
            poem_path += "/*."+self.poem_ext
            
        # find paths
        self.paths = glob.glob(poem_path, recursive = True)

        self.paths = [path.replace("\\", "/") for path in self.paths]
    
    def read_poem(self, path):
        """
        Reads a poem from a file.
        
        Parameters
        ----------
        path (str):
            Path to the poem file.
            
        Returns
        -------
        poem (str):
            The poem. If poem could not be read, an empty string is returned.
        """
        try:
            with open(path, 'r', encoding="utf8") as f:
                return f.read()
        except:
            print(f"Could not read {path}")
            return ""
        
    def get_example(self, example_index=None):
        example_index = example_index or random.randint(0, len(self.paths))
        example = self.paths[example_index]
        
        cat, title, author = extract_poem_info(example)
        print(f"File index: {example_index}")
        print(f"Category: {cat}")
        print(f"Title: {title}")
        print(f"Author: {author}")
        print("-"*(8+max([len(cat), len(title), len(author)])))
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
    # test extract poem info
    import glob, random
    paths = glob.glob("data/"+"kaggle_poems/topics/*/*.txt", recursive=True)
    example = random.choice(paths)
    print("Input path:", example)
    cat, title, author = extract_poem_info(example)
    print("Category:", cat)
    print("Title:", title)
    print("Author:", author)
    
    # test KagglePoems class
    poems = KagglePoems()
    poems.get_example()
    print(poems.stats)