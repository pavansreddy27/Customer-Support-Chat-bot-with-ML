# Customer-Support-Chat-bot-with-ML

## Overview
This project is a multi-domain chatbot built using Flask and machine learning models. It supports multiple applications, including shopping, social media, travel, payments, food ordering, jobs, and entertainment. Each application is powered by different machine learning models to handle user queries efficiently.

## Features
- **Shopping Chatbot**: Uses a Logistic Regression model trained on shopping FAQs.
- **Social Media Chatbot**: Utilizes a Naïve Bayes model trained on social media-related FAQs.
- **Travel Chatbot**: Uses a Logistic Regression model trained on travel-related queries.
- **Payments Chatbot**: Loads a pre-trained Logistic Regression model for handling payment-related queries.
- **Food Ordering Chatbot**: Uses Cosine Similarity to match user queries with food-related FAQs.
- **Jobs Chatbot**: Employs a Logistic Regression model trained on job-related queries.
- **Entertainment Chatbot**: Uses a Logistic Regression model to answer streaming-related questions.

## Tech Stack
- **Backend**: Flask (Python)
- **Machine Learning Models**: Logistic Regression, Naïve Bayes, Cosine Similarity
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Database**: TSV files and CSV files for storing FAQs and training data

## Installation
### Prerequisites
- Python 3.7+
- Flask
- Pandas
- Scikit-learn
- NLTK
- Joblib

## API Endpoints
- **`POST /get_response`** - Gets response from the shopping chatbot.
- **`POST /facebook_webhook`** - Fetches responses for social media queries.
- **`POST /semantic_search`** - Handles semantic search queries.
- **`POST /handle_travel_query`** - Provides responses for travel-related queries.
- **`POST /get_answer`** - Fetches responses from the entertainment chatbot.
- **`POST /get_chat_response`** - Handles payment-related queries.
- **`POST /get_food_chat_response`** - Provides responses for food ordering queries.
- **`POST /get_job_answer`** - Fetches job-related answers.

## Future Enhancements
- Improve chatbot responses with deep learning models.
- Integrate with a front-end UI.
- Expand support for more domains.


## Contributing
Feel free to fork this repository and submit pull requests for improvements.

## Contact
For any inquiries, contact Pavan S Reddy at [pavansreddy26.com].

