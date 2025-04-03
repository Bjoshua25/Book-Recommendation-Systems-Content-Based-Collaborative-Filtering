# **Book Recommendation Systems: Content-Based Filtering**  

![](image_cover.png)

## **INTRODUCTION**  
With the increasing volume of books available online, manually selecting relevant books becomes challenging. **Content-Based Filtering** is a personalized recommendation approach that suggests books similar to those a user has liked based on their features.  

This project implements a **content-based book recommendation system** using the **Goodbooks-10k dataset**.  

---

## **PROBLEM STATEMENT**  
Book lovers often struggle to find the next great read based on their preferences. The goal of this project is to:  
- **Build a recommendation system that suggests books similar to a given book.**  
- **Use natural language processing (NLP) techniques** to analyze book descriptions and metadata.  
- **Evaluate the effectiveness of the recommendations.**  

---

## **SKILL DEMONSTRATION**  
- **Text Data Processing (TF-IDF Vectorization)**  
- **Cosine Similarity for Recommendation**  
- **Content-Based Filtering Algorithm**  
- **Recommendation System Implementation**  

---

## **DATA SOURCING**  
The dataset used is **Goodbooks-10k**, which contains:  
- **Book Titles**  
- **Authors**  
- **Genres**  
- **User Ratings**  
- **Book Descriptions**  

---

## **DATA PREPROCESSING**  
- **Text Cleaning:** Removing special characters and stopwords from book descriptions.  
- **TF-IDF Vectorization:** Converting textual data into numerical form.  
- **Cosine Similarity Computation:** Measuring book similarity based on vectorized features.  

![](barchart.png)

## **MODELLING: CONTENT-BASED FILTERING**  
- **TF-IDF (Term Frequency-Inverse Document Frequency) Approach**  
- **Similarity Matrix using Cosine Similarity**  
- **Top N Book Recommendations for a Given Book**  

### **Example Code for Book Recommendation**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert book descriptions to TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['description'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend books
def get_recommendations(title, books_df, similarity_matrix):
    index = books_df[books_df['title'] == title].index[0]
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_books = [books_df.iloc[i[0]]['title'] for i in scores[1:6]]
    return top_books
```

---

## **ANALYSIS & VISUALIZATION**  
- **Word Cloud for Most Common Book Themes**  
- **Distribution of Ratings Across Books**  
- **Similarity Scores between Books**  

---

## **CONCLUSION**  
- **Content-based filtering successfully recommends books based on their descriptions.**  
- **TF-IDF and Cosine Similarity provide a structured way to analyze textual data.**  
- **The model can be expanded by integrating user preferences and collaborative filtering.**  
