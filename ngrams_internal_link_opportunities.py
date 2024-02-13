import requests
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string
from bs4 import BeautifulSoup, Comment
import spacy
# Dansk NLP-model loades
nlp = spacy.load('da_core_news_sm')
import concurrent.futures
import itertools
import time
import multiprocessing

###############################################################################
################################ URL IMPORT ###################################
###############################################################################

# URLs are imported
url_df = pd.read_excel(r'C:/your-path-to-url-file/url-file.xlsx')   

# URLs are saved in a list
urls = list(url_df['URL'])

# Remove all URLs that contain "lang="
urls = [x for x in urls if "lang=" not in x ]

###############################################################################
########################### CREATE FUNCTIONS ##################################
###############################################################################

# Function for scraping URLs
def fetch_url(url):
    
    # Counter to print progress
    with lock:
        url_counter.value += 1
        print(f"Fetched HTML from {url_counter.value}/{len(urls)} URLs")
    
    return requests.get(url).text

#_______________

# Function for extracting n-grams
def extract_ngrams(data, num, stop_words):
    # Tokenize the text
    tokens = word_tokenize(data)
    
    # Convert tokens to lowercase and exclude stopwords and punctuation
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token not in string.punctuation and ":" not in token and "–" not in token]
    
    # Create n-grams
    n_grams = ngrams(filtered_tokens, num)
    gram_list = [' '.join(grams) for grams in n_grams]
    
    return gram_list

#_______________

# Function for extracting visible text from website
def extract_visible_text(html_page):
    soup = BeautifulSoup(html_page, 'html.parser')
    
    # Exclude irrelevant HTML tags
    for script in soup(["script", "noscript", "nav", "style", "input", "meta", "label", "header", "footer", "aside", "h1", "a", "table", "button"]):
        script.decompose()

    # Remove various comments
    for element in soup.find_all(text=lambda text: isinstance(text, Comment)):
        element.extract()

    # Extract visible text
    page_text = soup.get_text()

    # Further cleaning of various characters
    page_text = page_text.lower().strip().replace("  ","")
    text_content = "".join([s for s in page_text.splitlines(True) if s.strip("\r\n")])
    
    return text_content

###############################################################################
############################# CREATE N-GRAMS ##################################
###############################################################################

# Start timer for n-grams section
time_ngrams_start = time.time()

# Danish stopwords are imported from GitHub
stop_words_da = list(set(pd.read_html('https://gist.github.com/berteltorp/0cf8a0c7afea7f25ed754f24cfc2467b')[0][1].tolist()))

# Option to add more stopwords
additional_stopwords = ["’","”"]
stop_words_da.extend(additional_stopwords)

#_______________

# Create counter for multiprocessing
url_counter = multiprocessing.Value('i', 0)
lock = multiprocessing.Lock()

# Extract text from URLs in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    html_pages = list(executor.map(fetch_url, urls))

#_______________

# Counter for counting iterations
current_iteration = 0

# Create an empty list
ngram_dedupe = []
# Loop for tokenizing and creating n-grams
for html_page, url in zip(html_pages, urls):
  # Extract text from URL
  text_content = extract_visible_text(html_page)

  # Tokenize text
  doc = nlp(text_content)
  words = [token.text for token in doc]
      
  # POS-tag tokens
  pos_tags = [(token.text, token.pos_) for token in doc]
  # Exclude irrelevant POS elements
  keywords = [x[0] for x in pos_tags if x[1] not in ['NUM', 'ADV', 'ADJ', 'ADP', 'AUX', 'DET', 'VERB', 'PRON', 'PUNCT','X', 'SPACE']]

  # Reassemble keywords into a single text and create n-grams from it
  keywords_bi_grams = extract_ngrams(" ".join(keywords), 2, stop_words_da)
  keywords_tri_grams = extract_ngrams(" ".join(keywords), 3, stop_words_da)
  
  # Merge bi_grams and tri_grams
  n_grams = keywords_bi_grams + keywords_tri_grams 
  # Add URL to the keyword list
  n_grams.append(url)
  # Add keywords and URL to the n-gram list
  ngram_dedupe.append(n_grams)
  
  # Print current iteration
  current_iteration += 1
  print(f"POS-tagged {current_iteration}/{len(urls)} page texts")

# End timer for n-grams section
time_ngrams_end = time.time()

# Time is calculated and printed
diff_ngrams = time_ngrams_end - time_ngrams_start
execution_ngrams = diff_ngrams / 60
print('Execution time for N-grams section:', round(execution_ngrams, 2), 'minutes')
###############################################################################
###################### URLS MATCHED ON N-GRAMS ###############################
###############################################################################

# Timer for matching section starts
time_matching_start = time.time()

# Column names and empty dataframe are created
column_names = {'from url': [], 'to url': [], 'n_gram matches':[]}
n_grams_matches_df = pd.DataFrame(data=column_names)

# Loop for n-grams comparison
results = []
for (x, item_url), (y, item_url2) in itertools.combinations(zip(ngram_dedupe, urls), 2):
    ngram_matches = set(x).intersection(y)
    results.append({"from url": item_url, "to url": item_url2, "n_gram matches": len(ngram_matches)})

n_grams_matches_df = pd.DataFrame(results)

#_______________

# Sorted by highest number of n-gram matches
n_grams_matches_df = n_grams_matches_df.sort_values(by=['n_gram matches'], ascending=False)

# All urls with fewer than 75 n-gram matches are excluded
n_grams_matches_df = n_grams_matches_df[n_grams_matches_df['n_gram matches'] >= 75].reset_index(drop=True)
#_______________

# Timer for matching section ends
time_matching_end = time.time()

# Time is calculated and printed
diff_matching = time_matching_end - time_matching_start
execution_matching = diff_matching / 60
print('Execution time for URL matching section:', round(execution_matching, 2), 'minutes')

#_______________
# The finished dataframe is exported to Excel
n_grams_matches_df.to_excel('C:/your-path/n_grams_matches_df.xlsx')
