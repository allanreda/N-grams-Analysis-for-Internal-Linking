# N-Grams Analysis for Internal Linking

## Overview
This script conducts an N-Grams analysis to improve internal linking between web pages by identifying matches of n-grams across different URLs. It's configured to analyze a specified list of URLs containing Danish texts only. The process involves cleaning html-content from each URL, extracting n-grams, and then matching URLs based on shared n-grams to suggest potential internal linking opportunities.

## Functionality
The script operates in several key steps: First it extracts text content from each URL and preprocesses this data by removing stopwords, punctuation, and performing other cleaning operations. It then generates n-grams from the cleaned text and deduplicates them to focus on unique content markers. The core functionality involves comparing these n-gram sets across URLs to find matches, suggesting where internal links could be beneficially created. The process ends by ranking these matches to prioritize linking suggestions and exporting the results to Excel for further analysis. The end result of the script is a DataFrame that look like this:  
![image](https://github.com/allanreda/N-grams-Analysis-for-Internal-Linking/assets/89948110/d900b57e-d071-424c-8dbc-06e1bd00f887)



## Technologies
The project is built using:  
-Beautiful Soup and requests for web scraping  
-multiprocessing and concurrent.futures for parallel processing the web scraping  
-NLTK for text processing and n-grams generation  
-The Danish language model da_core_news_sm for POS-tagging Danish text  
-Pandas for data management and manipulation  
-itertools for efficient looping through combinations of URLs

## Goal  
The primary goal is to enhance website SEO performance and user navigation through internal linking, based on content relevance defined by shared n-grams. This approach aims to automate and thereby make it easier to identify internal linking opportunities across large websites. By streamlining this aspect of SEO, the script aids in improving site structure, enhancing user experience, and potentially boosting search engine rankings. 
