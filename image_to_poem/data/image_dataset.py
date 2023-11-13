# CODE adapted from https://github.com/arthurdjn/img2poem-pytorch/

import requests
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def download_image(url, outname, outdir='./'):
    """Download an image from a URL.
    Original code from a `post <https://stackoverflow.com/a/30229298>`__ on Stackoverflow.

    Args:
        url (str): URL to the image.
        outname (str): Name of the image, with its extension.
        outdir (str): Path to the saving directory.

    Example:
        >>> url = "http://farm4.staticflickr.com/3910/14393814286_c6dbcf7a92_z.jpg"
        >>> outname = "my_image.png"
        >>> download_image(url, outname)
    """
    with open(os.path.join(outdir, outname), 'wb') as handle:
        try:
            response = requests.get(url, stream=True)
            for block in response.iter_content(1024):
                handle.write(block)
        except Exception as error:
            print(f"WARNING: An error occured. {error}"
                  f"Could not download the image {outdir}/{outname} from the URL {url}.")
    return os.path.join(outdir, outname)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # To float values between [0, 1]
    transforms.ToTensor(),
    # Normalize regarding ResNet training data
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PoemMultiMDataset(Dataset):
    r"""MultiM Poem Dataset with masks used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifier of the image & poem pair.

    * :attr:`tokens` (torch.tensor): Tokenized ids of a poem.

    * :attr:`masks` (torch.tensor): Tokenized ids masked.

    * :attr:`image` (torch.tensor): Matrix of the image in RGB format.

    .. note::
        The default filename used to process the data is called ``multim_poem.json``.
        The ``image_dir`` argument is used to locate the downloaded images.

    .. note::
        Download the images from the json file with the ``download`` class method.

    """

    url = 'https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json'
    dirname = 'img2poem'
    name = 'multim'

    def __init__(self, root = "data/images/", tokenizer=None, max_seq_len=128, transform=None):
        super(PoemMultiMDataset, self).__init__()
        # self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transform or DEFAULT_TRANSFORM
        # data = pd.read_json(self.__class__.filename)
        data = pd.read_json(self.url)
        ids = []
        poems = []
        images = []
        for _, row in tqdm(data.iterrows(), desc='Loading', position=0, leave=True, total=len(self.data)):
            id = row.id
            poem = row.poem.replace("\n", " ; ")
            image_file = os.path.join(root, f'{id}.jpg')
            try:
                image = self.transform(Image.open(image_file).convert('RGB'))
                ids.append(id)
                poems.append(poem)
                images.append(image)
            except Exception:
                pass

        # tokens, tokens_ids, masks = pad_sequences(poems, tokenizer,
                                                #   max_seq_len=max_seq_len,
                                                #   sos_token="[CLS]",
                                                #   eos_token="[SEP]",
                                                #   pad_token="[PAD]")
        self.ids = torch.tensor(ids)
        self.images = torch.stack(images)
        # self.tokens = tokens
        # self.tokens_ids = torch.tensor(tokens_ids)
        # self.masks = torch.tensor(masks)

    @classmethod
    def download(cls, root='.data', **kwargs):
        df = pd.read_json(cls.url)
        outdir = os.path.join(root, cls.dirname, cls.name)
        for _, row in tqdm(df.iterrows(), desc='Downloading', position=0, leave=True, total=len(df)):
            id = row.id
            url = row.image_url
            image_file = os.path.join(outdir, f'{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
            except Exception:
                print(f"WARNING: Image {id} not downloaded from {url}.")

        return PoemMultiMDataset(outdir, **kwargs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.images[index]#self.tokens_ids[index], self.masks[index], self.images[index]
    
    
if __name__ == "__main__":
    dataset = PoemMultiMDataset.download()
    print(len(dataset))