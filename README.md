# Fetch_Solution
NLP and Intelligent search of the most relevant offers to an input query using Latent Semantic Indexing.

The project aims to create a master dataset using outer join from three CSV files using MySQL Workbench and then perform semantic search using Latent Semantic Indexing (LSI) along with various text analytics techniques. The final dataset consists of five columns: OFFER, RETAILER, BRAND, BRAND_BELONGS_TO_CATEGORY, and IS_CHILD_CATEGORY_TO. The goal is to simplify query building and improve search results by incorporating LSI.

Download the Fetch_solution4.ipynb and FINAL.csv file and run the ipynb file. Both the csv and ipynb fies are to be in same folder.
Install these modules
numpy,
pandas,
spacy,
python -m spacy download en_core_web_sm,
gensim

Acceptance Criteria
If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category.
If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand.
If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer.
The tool also returns the relevance score that was used to measure the similarity of the text input with each offer.

