# Fetch_Solution
NLP and Intelligent search of the most relevant offers to an input query using Latent Semantic Indexing.

The project aims to create a master dataset from three CSV files using MySQL Workbench and then perform semantic search using Latent Semantic Indexing (LSI) along with various text analytics techniques. The final dataset consists of five columns: OFFER, RETAILER, BRAND, BRAND_BELONGS_TO_CATEGORY, and IS_CHILD_CATEGORY_TO. The goal is to simplify query building and improve search results by incorporating LSI.


Acceptance Criteria
If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category.
If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand.
If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer.
The tool also returns the relevance score that was used to measure the similarity of the text input with each offer.

