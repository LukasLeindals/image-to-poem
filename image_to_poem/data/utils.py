import re

def extract_poem_info(path):
    """
    Extracts the poem info from the path of a poem file
    
    Args:
        path (str): The path to the poem file
    
    Returns:
        (str, str, str): The topic, title, and author of the poem
    """
    # ensure path is in unix format
    path = path.replace("\\", "/")
    
    # get topic
    topic = path.split('/')[-2]
    
    # get title and author
    split_on_capitals = lambda x: re.sub(r"(\w)([A-Z])", r"\1 \2", x)
    name = path.split('/')[-1].replace('.txt', '')
    title, author = name.split('Poemby')
    title, author = split_on_capitals(title), split_on_capitals(author)
    
    # remove first two words from title as these describe the topic
    title = ' '.join(title.split(' ')[2:])
    
    return topic, title, author



if __name__ == "__main__":
    import glob, random
    paths = glob.glob("data/"+"kaggle_poems/topics/*/*.txt", recursive=True)
    example = random.choice(paths)
    print("Input path:", example)
    topic, title, author = extract_poem_info(example)
    print("Topic:", topic)
    print("Title:", title)
    print("Author:", author)

    # def print_example(example):
    #     print(f"Topic: {example.split('/')[-2]}")
    #     split_on_capitals = lambda x: re.sub(r"(\w)([A-Z])", r"\1 \2", x)
    #     name = example.split('/')[-1].replace('.txt', '')
    #     title, author = name.split('Poemby')
    #     title, author = split_on_capitals(title), split_on_capitals(author)
    #     print(f"Title: {title}")
    #     print(f"Author: {author}")
    #     print("-"*(8+max([len(topic), len(title), len(author)])))
    #     with open(example, 'r') as f:
    #         print(f.read())