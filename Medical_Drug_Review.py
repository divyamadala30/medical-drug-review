import streamlit as st

st.markdown("## Final Project for Practical Applications of Machine Learning (PAML)")

st.markdown(""" Medical Drug Review Dataset

This involves training and evaluating ML end-to-end pipeline in a web application using the Medical Drug Reviews dataset.
 
We are using classification machine learning methods such as Logistic Regression and Stochas- tic Gradient Descent, along with preprocessing techniques like n-gram, Term-Frequency Inverse Document Frequency (TF-IDF), and bag-of-words. These methods are well- suited for sentiment analysis tasks and are commonly used in text classification tasks.

1. Drug (categorical): name of drug
2. DrugId (numerical): drug id
3. Condition (categorical): name of condition
4. Review (text): patient review
5. Side (text): side effects associated with drug (if any)
6. EaseOfUse (numerical): 5 star rating
7. Effectiveness (numerical): 5 star rating
8. Satisfaction (numerical): 5 star rating
9. Date (date): date of review entry
10. UsefulCount (numerical): number of users who found review useful.
11. Age (numerical): age group range of user
12. Sex (categorical): gender of user
""")

st.markdown("Click **Preprocess Dataset** to get started.")