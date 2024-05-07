import pandas as pd                   
import streamlit as st                 
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import plotly.express as px

st.markdown('# Explore & Preprocess Dataset')

webmddf = pd.read_csv("datasets/webmd.csv")
st.session_state['data'] = webmddf

# def plot_scatterplot(df, feature1, feature2):
#     fig = px.scatter(data_frame=df, x=feature1, y=feature2,
#                       range_x=[df[feature1].min(), df[feature1].max()],
#                       range_y=[df[feature2].min(), df[feature2].max()])
#     return fig

#st.write(plot_scatterplot(webmddf, 'Satisfaction', 'Drug'))

#cleaning dataset
relevant_cols=['Reviews','Satisfaction','DrugId']
webmddf = webmddf.loc[:, relevant_cols]

webmddf.dropna(subset=['Reviews','Satisfaction','DrugId'], inplace=True)
webmddf.reset_index(drop=True, inplace=True)

st.markdown("### Before cleaning the dataset")
st.write(webmddf.head())

def remove_punctuation(text):
    try: # python 2.x
        text = text.translate(None, string.punctuation) 
    except: # python 3.x
        translator = text.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    return text

webmddf['Reviews'] = webmddf['Reviews'].apply(remove_punctuation)
st.markdown("### After cleaning the dataset")
st.session_state['data'] = webmddf
st.write(webmddf.head())

# Plot Histogram
hist = px.histogram(webmddf, "Satisfaction", range_x=[1,5])
st.markdown("### Visualize features")
st.write(hist)

#stop words
stop_words_list = open("datasets/stopwords.txt","r")
stop_words_list = stop_words_list.readlines()
stop_words=[]
for word in stop_words_list:
    stop_words.append(word.split('\n')[0])

# feature encoding selection
st.markdown("### Select features")
feature_encoding = st.selectbox(
            'Select feature encoding word count',
            ["word count", "tfidf word count"])
st.session_state['feature_encoding'] = feature_encoding


# balance labels selection
balance_labels=st.selectbox(
            'Select if you to want to balance labels',
            ["False", "True"])
st.session_state['balance_labels'] = balance_labels


# analyzer selection
analyzer = st.selectbox(
            'Select analyzer',
            ["word", "char", "char_wb"])
st.session_state['analyzer'] = analyzer

# ngram range selection
ngram_range_selection = st.selectbox(
            'Select ngram range',
            ["unigram", "bigram", "unigram & bigram"])
if ngram_range_selection == "unigram":
    ngram_range=(1, 1)
if ngram_range_selection == "bigram":
    ngram_range=(2, 2)
if ngram_range_selection == "unigram & bigram":
    ngram_range=(1, 2) #unigram & bigram - (1,2)

st.session_state['ngram_range'] = ngram_range

# stop words
# stop_words = st.selectbox(
#             'Select stop words',
#             ["stop_words", "english"])
# if stop_words == "english":
#     st.session_state['stop_words'] = stop_words
# else:
#     st.session_state['stop_words'] = stop_words_list
stop_words = "english"
st.session_state['stop_words'] = stop_words


def word_count(df, analyzer, ngram_range, stop_words):
    count_vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)
    X_train_counts = count_vect.fit_transform(df['Reviews'])

    wc_feature_names = np.array(count_vect.get_feature_names_out())
    #print('wc_feature_names: {}'.format(wc_feature_names))

    word_count_df = pd.DataFrame(X_train_counts.toarray())
    word_count_df = word_count_df.add_prefix('word_count_')

    df = pd.concat([df, word_count_df], axis=1)
    #st.session_state['data'] = webmddf
    return df, wc_feature_names, word_count_df

def tfidf_word_count(df, analyzer, ngram_range, stop_words):
    tfidf_transformer = TfidfTransformer()
    count_vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)

    X_train_counts = count_vect.fit_transform(df['Reviews'])
    wc_feature_names = np.array(count_vect.get_feature_names_out())
    #print('wc_feature_names: {}'.format(wc_feature_names))

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    word_count_df = pd.DataFrame(X_train_tfidf.toarray())
    word_count_df = word_count_df.add_prefix('tf_idf_word_count_')
    df = pd.concat([df, word_count_df], axis=1)
    #st.session_state['data'] = webmddf
    return df, wc_feature_names, word_count_df

encoder = None
if (st.button('Run encoder')):
    if feature_encoding == "word count":
        encoder = word_count(webmddf, analyzer, ngram_range, list(stop_words))
    else:
        encoder = tfidf_word_count(webmddf, analyzer, ngram_range, list(stop_words))
    webmddf = encoder[0]
    st.session_state['data'] = webmddf
    st.write(encoder[0])

if encoder is not None:
    # tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)

    # X_train_tfidf = tfidf.fit_transform(webmddf['Reviews'])
    # tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
    # # encoder[2] is word_count_df
    # tfidf_df = encoder[2].add_prefix('tfidf_word_count_')
    # tfidf_feature_names = np.array(tfidf.get_feature_names_out())
    # products = pd.concat([webmddf, tfidf_df], axis=1)

    # new_doc = webmddf['Reviews'][:2].to_numpy()
    # responses = tfidf.transform(new_doc)

    tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=list(stop_words))

    X_train_tfidf = tfidf.fit_transform(webmddf['Reviews'])
    tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
    word_count_df = pd.DataFrame(X_train_tfidf.toarray())
    word_count_df = word_count_df.add_prefix('tf_idf_word_count_')
    tfidf_df = word_count_df.add_prefix('tfidf_word_count_')
    tfidf_feature_names = np.array(tfidf.get_feature_names_out())
    print('tfidf_feature_names: {}'.format(tfidf_feature_names))

    webmddf = pd.concat([webmddf, tfidf_df], axis=1)
    print(webmddf.head())

    def get_top_tf_idf_words(response, feature_names, top_n=3):
        sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
        return feature_names[response.indices[sorted_nzs]]

    top_n=10
    new_doc = webmddf['Reviews'][:top_n].to_numpy()
    review_sample = tfidf.transform(new_doc)
    # encoder[1] is wc_feature_names
    review_top_words = get_top_tf_idf_words(review_sample, encoder[1], top_n=top_n)

    st.markdown("### These are the top words in Reviews column")
    st.write(review_top_words)

webmddf = webmddf[webmddf['Satisfaction'] != 3]
webmddf.reset_index(drop=True, inplace=True)
st.write(webmddf.shape)

st.session_state['data'] = webmddf