import zipfile
from tqdm import tqdm
import os
import glob
import json
import requests
import time
import torch
from transformers import pipeline
import datetime

from image_to_poem.utils import load_json_file, save_json_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ROOT = "data/"

def extract_kaggle(root = DEFAULT_ROOT, filter = "topics"):
    """
    Extracts the poems from the kaggle_poems.zip file into the kaggle_poems folder
    
    Parameters
    ----------
    root (str): 
        The root directory to extract the poems to
    filter (str): 
        The folder to extract from the zip file, None for all
    """
    # ensure file exists
    if not os.path.exists(os.path.join(root, "kaggle_poems.zip")):
        raise FileNotFoundError("Could not find 'kaggle_poems.zip' in the data folder, please download it from https://www.kaggle.com/datasets/michaelarman/poemsdataset/")
    
    # ensure root ends with a slash
    if not root.endswith("/"):
        root += "/"
    
    # init unfound paths
    unfound_kaggle_poems_paths = []
    
    with zipfile.ZipFile(root+"kaggle_poems.zip", 'r') as zip_ref:
        # get all file names
        files = zip_ref.namelist()
        
        # filter to only get the files in the topics folder
        if filter is not None:
            files = [file for file in files if file.startswith(f"{filter}/")]

        # extract
        num_extracted = 0
        for file in tqdm(files, desc="Extracting Kaggle Poems"):
            try:
                zip_ref.extract(file, root + "kaggle_poems/")
                num_extracted += 1
            except FileNotFoundError:
                print(f"Could not find '{file}'")
                unfound_kaggle_poems_paths.append(file)
    

    print(f"Extracted {num_extracted} poems from Kaggle")
    
def download_images(max_imgs = 100, root = os.path.join(DEFAULT_ROOT, "poem_images")):
    """
    Downloads images from the multim_poem.json file
    
    Parameters
    ----------
    max_imgs (int): 
        The maximum number of images to download
    root (str): 
        The root directory to save the images to
    """
    
    # make directories
    os.makedirs(root, exist_ok=True)
    
    # get json
    json_path = "data/multim_poem.json"
    if not os.path.exists(json_path):
        json_url = "https://raw.githubusercontent.com/arthurdjn/img2poem-pytorch/master/data/poems/multim_poem.json"
        with open(json_path, 'w') as f:
            json.dump(requests.get(json_url).json(), f)
    
    image_json = json.load(open(json_path))
    
    # download images
    num_downloaded = 0
    for poem in tqdm(image_json, desc="Downloading Images"):
        # extract info
        image_url = poem["image_url"]
        image_id = poem["id"]
        image_path = os.path.join(root, f"{image_id}.jpg")
        
        # download image if it doesn't exist
        if not os.path.exists(image_path):
            try:
                with open(image_path, 'wb') as handle:
                    response = requests.get(image_url, stream=True)
                    for block in response.iter_content(1024):
                        handle.write(block)
                num_downloaded += 1
            except Exception:
                print(f"Could not download image {image_id} from {image_url}")
                
            # sleep every 10 images to avoid getting blocked
            if num_downloaded % 10 == 0:
                time.sleep(2)
        else:
            num_downloaded += 1
        
        # break if we have enough images
        if num_downloaded >= max_imgs:
            break
                
    print(f"Downloaded {num_downloaded} images")
    

def create_caption_data(N : int = None, image2text = None, save=True, verbose = False, backup_every = 100):
    """
    Creates a json file with the image urls, ids and captions.
    
    Parameters
    ----------
    N (int, optional): 
        The maximum number of captions to create, if None all will be used. Defaults to None.
    image2text (pipeline, optional):
        The pipeline to use for creating the captions, if None "nlpconnect/vit-gpt2-image-captioning" will be initialized and used. Defaults to None.
    save (bool, optional):
        Whether to save the data as a json file. Defaults to True.
    verbose (bool, optional):
        Whether to print verbose output. Defaults to False.
    backup_every (int, optional):
        How often to save a backup of the data. Defaults to 100.
        
    Returns
    -------
    data (list of dict):
        A list of dicts containing the image urls, ids and captions.
    """
    # set paths
    DATAPATH = DEFAULT_ROOT + "multim_poem.json"
    PROCESSED_DATAPATH = DEFAULT_ROOT + "caption_poem.json"
    BACKUP_DATAPATH = DEFAULT_ROOT + "backups/" + "caption_poem.json"
    os.makedirs(DEFAULT_ROOT + "backups/", exist_ok=True)
    
    # read data
    all_data = load_json_file(DATAPATH)
    
    # create image_to_text function
    if image2text is None:
        image2text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=DEVICE)
    
    # create variables
    N = N if N is not None else len(all_data)    
    data = [{}]*N
    i,j = 0,0 # i = img indexes, j = succesful downloads
    
    # add captions
    while j < N and i < len(all_data):
        t0 = time.time()
        try:
            # get image description from model
            desc = image2text(all_data[i]['image_url'])
            if verbose:
                print(f"Outcome of creating caption for image {all_data[i]['id']}: succeeded in {time.time()-t0:.3f} s")
        except:
            # Image could not be read, skip 
            if verbose:
                print(f"Outcome of creating caption for image {all_data[i]['id']}: failed in {time.time()-t0:.3f} s")
            i += 1
            continue
        desc = desc[0]['generated_text']
        
        # copy data
        data[j] = all_data[i]
        # add description to the data
        data[j]["caption"] = desc
        
        # save current progreess to backup file
        if save and i % backup_every == 0:
            save_json_file(BACKUP_DATAPATH, data[:j])
        
        # update 
        i += 1 
        j += 1 
    
    # print status 
    print(f"Datapoints avail : {len(all_data)}\nDatapoints used  : {j}")
    
    # remove empty dicts
    data = data[:j]
    
    # save data as a json file 
    if save:
        save_json_file(PROCESSED_DATAPATH, data)
    
    return data

                

if __name__ == "__main__":
    # extract_kaggle()
    # download_images()
    
    # create caption data
    image2text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=DEVICE)
    t0 = time.time()
    N = None
    create_caption_data(N, image2text=image2text, save = True, verbose = True, backup_every=100)
    print(f"Captions created in {time.time()-t0:.3f} s")
    
        