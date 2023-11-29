import numpy as np
from image_to_poem.data.prep_data import load_json_file
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

def plot_hists():
    # read data 
    data = load_json_file("data/caption_poem.json")
    # allocate memory
    caption_len = [0]*len(data)
    poem_len = [0]*len(data)
    # collect information 
    for i,data_point in tqdm(enumerate(data)):
        p = word_tokenize(data_point["poem"])
        c = word_tokenize(data_point["caption"])
        caption_len[i] = len(c)
        poem_len[i] = len(p)
    
    # plot 
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    sns.histplot(caption_len, ax=axs[0])
    sns.histplot(poem_len, ax=axs[1])
    titles = ["Caption", "Poem"]
    for i,ax in enumerate(axs):
        ax.set_xlabel("$\#$Tokens")
        ax.set_title(f"{titles[i]} Lengths")
    plt.show()
    
    # print information:
    max_comb = np.max(caption_len) + np.max(poem_len) + 3
    print("upper bound length =", max_comb)

plot_hists()
