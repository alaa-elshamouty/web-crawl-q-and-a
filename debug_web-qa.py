import json
import os
import re
import urllib.request
from collections import deque
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from hyperlink_parser import HyperlinkParser

# Use a pipeline as a high-level helper

# pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

# # Load the model and tokenizer of the generator model

# tokenizer = AutoTokenizer.from_pretrained("C://Users/Alaa/.llama/checkpoints/Llama3.1-8B/")
# model = AutoModelForCausalLM.from_pretrained("C://Users/Alaa/.llama/checkpoints/Llama3.1-8B/")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")

# Load the tokenizer and model of the retrieval model
embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
embedding_model = AutoModel.from_pretrained("BAAI/bge-large-en")

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Define root domain to crawl
domain = "porsche.com/stories"
full_url = "https://www.porsche.com/stories/experience/road-trip-in-a-porsche-944/"


# "https://www.porsche.com/stories/"

# Create a class to parse the HTML and get the hyperlinks



# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks


# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


def crawl(url, max_nr_urls=10):
    # Parse the URL and get the domain
    local_domain = domain  # urlparse(url).netloc
    local_domain_filename = urlparse(local_domain).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain_filename + "/"):
        os.mkdir("text/" + local_domain_filename + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
        os.mkdir("processed")
    nr_urls = 0
    # While the queue is not empty, continue crawling
    while queue and nr_urls < max_nr_urls:

        # Get the next URL from the queue
        url = queue.pop()
        print(url)  # for debugging and to see the progress
        filename = re.sub(r'[<>:"/\\|?*]', '_', url[8:])

        # your code to write text to file goes here
        # Save text from the url to a <url>.txt file

        # Get the text from the URL using BeautifulSoup
        soup = BeautifulSoup(requests.get(url).text, "html.parser")

        # Remove script and style content
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Extract title
        title = soup.title.string if soup.title else "No Title"

        # List of potential classes that may contain section titles
        title_classes = ["hero__category", "taxonomy__title", "section-title", "header-title"]
        # TODO: check why the section title is None
        content = extract_content(soup, title_classes)

        # Store in JSON format
        structured_data = {
            "title": title,
            "url": url,
            "content": content
        }
        if len(structured_data["content"]) > 0:
            # get total number of words over sections
            print("Saving to file: ", filename + ".json" + "\n",
                  "\t number of words:{}".format(total_words_over_sections(structured_data)) + "\n",
                  "\t number of sections:{}".format(len(structured_data["content"])) + "\n",
                  "\t words per section:{}".format(
                      {"s{}".format(i): len(n["text"].split()) for i, n in
                       enumerate(structured_data["content"])}) + "\n",
                  )
            # Save to a JSON file
            # with open('text/'+local_domain+'/'+filename + ".txt", "w") as f
            with open('text/' + local_domain_filename + '/' + filename + ".json", "w", encoding="utf-8") as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=4)
            nr_urls += 1
        else:
            print("\t No content found for: ", url)
        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

def embed_text(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
    return embeddings.cpu().numpy() if embeddings.is_cuda else embeddings.numpy()

def total_words_over_sections(structured_data):
    return sum([len(section["text"].split()) for section in structured_data["content"]])

def get_chunks(text, max_tokens=500):
    if len(embedding_tokenizer.encode(" " + text)) <= max_tokens:
        return [text]
    sentences = text.split('. ')
    n_tokens = [len(embedding_tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks, tokens_so_far, chunk = [], 0, []
    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk, tokens_so_far = [], 0
        if token > max_tokens:
            continue
        chunk.append(sentence)
        tokens_so_far += token + 1
    if chunk:
        chunks.append(". ".join(chunk) + ".")
    return chunks

def create_context(question, df, max_len=2000):
    question_embedding = torch.tensor(embed_text(question))
    # Ensure each embedding in the DataFrame is converted to a tensor
    embedding_tensors = [torch.tensor(embedding) for embedding in df['embeddings'].values]

    # Stack them to form a single tensor for easier cosine similarity calculation
    embeddings_tensor = torch.stack(embedding_tensors)
    df['distances'] = torch.nn.functional.cosine_similarity(
        embeddings_tensor,
        question_embedding,
        dim=1
    )
    returns, cur_len = [], 0
    for _, row in df.nsmallest(df.shape[0], 'distances').iterrows():
        cur_len += len(embedding_tokenizer.encode(" " + row['text'])) + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])
    return "\n\n###\n\n".join(returns)

def answer_question(df, question, max_len=512, max_tokens=150, stop_sequence=None, debug=False):
    context = create_context(question, df, max_len=max_len)
    if debug:
        print("\t Context used :\n" + context)
        print("\n\n")
    prompt = (
        f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"
        f"Context: {context}\n\n"
        f"---\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    output = model.generate(
        inputs["input_ids"].to('cuda'),
        max_length=512,  # Try lowering to reduce latency
        #num_beams=2,  # Reduced beams for faster generation
        #no_repeat_ngram_size=2
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    return answer

def create_df_from_json_files(directory):
    """
    Create a DataFrame from the JSON files in the specified directory
    """
    # Get the list of JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    # Create a list to store the data
    data = []

    # Iterate over the JSON files and load the data
    for file in json_files:
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            data_pre_format = json.load(f)
            for i, content in enumerate(data_pre_format["content"]):
                data.append({
                    "url": data_pre_format["url"],
                    "title": data_pre_format["title"],
                    "section_title": content["section_title"],
                    "text": content["text"],
                    "embeddings": np.array(content["embedding"])
                })

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    return df


# crawl(full_url, max_nr_urls=1)
df = create_df_from_json_files("text")
question = "What day is it?"
answer = answer_question(df, question=question, debug=True, max_len=256)
print('*Question:', question,
      '\n Answer:', answer)
# now we have chunks of text but without correct section titles.
# next steps:
# 1. extract correct section titles and fix saving their chunk in the json file
# 2. check what kind of text preprocessing is needed
# 3. try asking questions relevant to those chunks (very naive rag)
#   - hf model takes forever to generate a response. Even with a simple prompt.
# 4. create question answer context pipeline to train embedding model
