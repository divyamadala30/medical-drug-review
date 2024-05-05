import pandas as pd                   
import streamlit as st                 
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from helper_functions import fetch_dataset, clean_data, display_review_keyword

st.markdown('# Explore & Preprocess Dataset')

def fetch_dataset():
    # Check stored data
    df = None
    data = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        data = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'])
        data = "/Users/divyamadala/Desktop/PAML/medical-drug-review/datasets/webmd.csv"
        if (data):
            df = pd.read_csv(data)
    if df is not None:
        st.session_state['data'] = df
    return df

def clean_data(df):
    data_cleaned = False
    # Drop irrelevant columns
    st.write('df.columns: {}'.format(df.columns))
    relevant_cols = ["Drug", "Satisfaction", "Reviews"]
    df = df.loc[:, relevant_cols]

    # Drop Nana
    df.dropna(subset=["Drug", "Satisfaction", "Reviews"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    #df.head()
    data_cleaned = True

    # Store new features in st.session_state
    st.session_state['data'] = df
    return df, data_cleaned

df = None
df = fetch_dataset()

def remove_punctuation(df, features):
    def remove_punctuation_2(text):
        try: # python 2.x
            text = text.translate(None, string.punctuation) 
        except: # python 3.x
            translator = text.maketrans('', '', string.punctuation)
            text = text.translate(translator)
        return text
    
    for i in features:
        df[i] = df[i].apply(remove_punctuation_2)
    # Confirmation statement
    st.write('Punctuation was removed from {}'.format(features))
    return df

def display_review_keyword(df, keyword, n_reviews=5):
    keyword_df = df['Reviews'].str.contains(keyword)
    filtered_df = df[keyword_df]#.head(n_reviews)

    return filtered_df

def word_count_encoder(df, feature, analyzer='word', ngram_range=(1, 1), stop_words=None):
    # Create CountVectorizer object using analyzer, ngram_range, and stop_words
    count_vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)
    # Add code here
    X_train_counts = count_vect.fit_transform(df[feature])

    word_count_df = pd.DataFrame(X_train_counts.toarray())
    word_count_df = word_count_df.add_prefix('word_count_')
    df = pd.concat([df, word_count_df], axis=1)
    # Store new features in st.session_state
    st.session_state['data'] = df
    print(X_train_counts.shape)
    return df, count_vect, word_count_df

def tf_idf_encoder(df, feature, analyzer='word', ngram_range=(1, 1), stop_words=None, norm=None):
    count_vect=None
    tfidf_transformer=None
    tfidf_df=None
    # Create CountVectorizer object using analyzer, ngram_range, and stop_words
    # Add code here
    count_vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)
    # Create TfidfTransformer object
    # Add code here
    tfidf_transformer = TfidfTransformer(norm=norm)
    X_train_counts = count_vect.fit_transform(df[feature])
    wc_feature_names = np.array(count_vect.get_feature_names_out())
    print('wc_feature_names: {}'.format(wc_feature_names))

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
    tfidf_df = tfidf_df.add_prefix('tf_idf_word_count_')
    df = pd.concat([df, tfidf_df], axis=1)

    # Show confirmation statement
    st.write(
        'Feature {} has been TF-IDF encoded from {} reviews.'.format(feature, len(tfidf_df)))

    # Store new features in st.session_state
    st.session_state['data'] = df

    return df, count_vect, tfidf_transformer, tfidf_df

########################### fetch dataset ################################
if df is not None:

    # Display original dataframe
    st.markdown('You have uploaded the Medical Drug Reviews dataset. This provides user reviews on specific drugs along with related conditions, side effects, age, sex, and satisfaction rating reflecting overall patient satisfaction.')

    st.markdown('View initial data with missing values or invalid inputs')
    st.dataframe(df)

    # Remove irrelevant features
    df, data_cleaned = clean_data(df)
    if (data_cleaned):
        st.markdown('The dataset has been cleaned. Your welcome!')
    
    ############## Task 1: Remove Punctation
    st.markdown('### Remove punctuation from features')
    removed_p_features = st.multiselect(
        'Select features to remove punctuation',
        df.columns,
    )
    if (removed_p_features):
        df = remove_punctuation(df, removed_p_features)
        # Store new features in st.session_state
        st.session_state['data'] = df
        # Display updated dataframe
        st.dataframe(df)
        st.write('Punctuation was removed from {}'.format(removed_p_features))

    # Use stopwords or 'engligh'
    st.markdown('### Use stop words or not')
    use_stop_words = st.multiselect(
        'Use stop words?',
        ['Use stop words', 'english'],
    )
    st.write('You selected {}'.format(use_stop_words))
    
    stop_words_list=[]
    st.session_state['stopwords']='english'
    if('Use stop words' in use_stop_words):
        stopwords_file = st.file_uploader(
            'Upload stop words file', type=['csv', 'txt'])
        # Read file
        if(stopwords_file):
            stop_words_df = pd.read_csv(stopwords_file)
            #stop_words = list(np.array(stop_words_df.to_numpy()).reshape(-1))
            # Save stop words to session_state
            st.session_state['stopwords'] = list(np.array(stop_words_df.to_numpy()).reshape(-1))
            st.write('Stop words saved to session_state.')
            st.table(stop_words_df.head())

    if('english' in use_stop_words):
        st.session_state['stopwords'] = 'english'
        st.write('No stop words saved to session_state.')


    # Inspect Reviews
    st.markdown('### Inspect Reviews')

    review_keyword = st.text_input(
        "Enter a keyword to search in reviews",
        key="review_keyword",
    )

    # Display dataset
    #st.dataframe(df)

    if (review_keyword):
        displaying_review = display_review_keyword(df, review_keyword)
        st.write('Summary of search results:')
        st.write('Number of reviews: {}'.format(len(displaying_review)))
        st.write(displaying_review)

    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)
    word_encoder = []

    # Initialize word encoders in session state
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = {'word_count':None, 'tfidf':None}
    st.session_state['tfidf_transformer'] = None

    word_count_col, tf_idf_col = st.columns(2)

    wc_analyzer = 'word'
    wc_n_ngram = (1,1)
    n_gram = {'unigram': (1,1), 'bigram': (2,2), 'unigram-bigram': (1,2)}

    ############## Task 2: Perform Word Count Encoding
    with (word_count_col):
        text_feature_select_int = st.selectbox(
            'Select text features for encoding word count',
            string_columns,
        )
        st.write('You selected feature: {}'.format(text_feature_select_int))

        wc_analyzer = st.selectbox(
            'Select the analyzer for encoding word count',
            ['word', 'char', 'char_wb'],
        )
        st.write('You selected analyzer: {}'.format(wc_analyzer))

        wc_n_ngram = st.selectbox(
            'Select n-gram for encoding word count',
            ['unigram', 'bigram', 'unigram-bigram'],
        )
        st.write('You selected n-gram: {}'.format(wc_n_ngram))

        if (text_feature_select_int and st.button('Word Count Encoder')):
            df, wc_count_vect, word_count_df = word_count_encoder(df, text_feature_select_int, analyzer=wc_analyzer, ngram_range=n_gram[wc_n_ngram], stop_words=st.session_state['stopwords'])
            word_encoder.append('Word Count')
            st.session_state['word_encoder'] = word_encoder
            st.session_state['count_vect']['word_count'] = wc_count_vect
            st.session_state['word_count_df']=word_count_df

            if('tfidf_word_count_df' in st.session_state):
                tfidf_word_count_df = st.session_state['tfidf_word_count_df']
                df = pd.concat([df, word_count_df, tfidf_word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df
            else:
                df = pd.concat([df, word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df

    ############## Task 3: Perform TF-IDF Encoding
    with (tf_idf_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for encoding TF-IDF',
            string_columns,
        )
        st.write('You selected feature: {}'.format(text_feature_select_onehot))

        tfidf_analyzer = st.selectbox(
            'Select the analyzer for encoding tfidf count',
            ['word', 'char', 'char_wb'],
        )
        st.write('You selected analyzer: {}'.format(tfidf_analyzer))

        tfidf_n_ngram = st.selectbox(
            'Select n-gram for encoding tfidf count',
            ['unigram', 'bigram', 'unigram-bigram'],
        )
        st.write('You selected n-gram: {}'.format(tfidf_n_ngram))

        if (text_feature_select_onehot and st.button('TF-IDF Encoder')):
            df, tfidf_count_vect, tfidf_transformer, tfidf_word_count_df = tf_idf_encoder(df, text_feature_select_onehot, analyzer=tfidf_analyzer, ngram_range=n_gram[tfidf_n_ngram], stop_words=st.session_state['stopwords'])
            word_encoder.append('TF-IDF')
            st.session_state['word_encoder'] = word_encoder
            st.session_state['count_vect']['tfidf'] = tfidf_count_vect
            st.session_state['transformer'] = tfidf_transformer
            st.session_state['tfidf_word_count_df']=tfidf_word_count_df

            if('word_count_df' in st.session_state):
                word_count_df = st.session_state['word_count_df']
                df = pd.concat([df, word_count_df, tfidf_word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df
            else:
                df = pd.concat([df, word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df

    # Display dataset
    st.dataframe(df)
    
    # Save dataset in session_state
    st.session_state['data'] = df

