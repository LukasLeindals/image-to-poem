import zipfile
from tqdm import tqdm
import os
import glob


class Data:
    root = "data/"
    
    def __init__(self, verbose = False) -> None:
        self.verbose = verbose
        
        # set paths
        self.set_paths()
    
    
    def set_paths(self):
        self.kaggle_poems_paths = glob.glob(os.path.join(self.root, "kaggle_poems/topics/*/*.txt"), recursive=True)
        self.num_kaggle_poems = len(self.kaggle_poems_paths)
    
    def extract_kaggle(self):
        if not os.path.exists(os.path.join(self.root, "kaggle_poems.zip")):
            raise FileNotFoundError("Could not find 'kaggle_poems.zip' in the data folder, please download it from https://www.kaggle.com/datasets/michaelarman/poemsdataset/")
        
        unfound_kaggle_poems_paths = []
        
        with zipfile.ZipFile(os.path.join(self.root, "kaggle_poems.zip"), 'r') as zip_ref:
            # get all file names
            files = zip_ref.namelist()
            
            # filter to only get the files in the topics folder
            files = [file for file in files if file.startswith("topics/")]
            
            # download
            for file in tqdm(files, desc="Downloading Kaggle Poems"):
                try:
                    zip_ref.extract(file, os.path.join(self.root, "kaggle_poems/"))
                except FileNotFoundError:
                    print(f"Could not find '{file}'")
                    unfound_kaggle_poems_paths.append(file)
        
        # set paths
        self.set_paths()
        print(f"Extracted {len(self.kaggle_poems_paths)} poems from Kaggle")
        
                

if __name__ == "__main__":
    data = Data()
    # data.extract_kaggle()
    # print(data.kaggle_poems_paths)
    print(data.num_kaggle_poems)