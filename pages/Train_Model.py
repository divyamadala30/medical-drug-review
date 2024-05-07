import numpy as np                    
from sklearn.model_selection import train_test_split
import streamlit as st                  
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

webmddf = st.session_state['data']

webmddf = webmddf[webmddf['Satisfaction'] != 3]
webmddf.reset_index(drop=True, inplace=True)
st.write("The shape of the dataframe is " + str(webmddf.shape))

webmddf['sentiment'] = webmddf['Satisfaction'].apply(lambda r: +1 if r > 3 else -1)
st.markdown("### View data")
st.write(webmddf.head())

st.markdown("### View number of positive and negative reviews")
st.write("Column Sentiment only refers to a review as positive(+1) or negative(-1)")
hist = px.histogram(webmddf, "sentiment", range_x=[-1,1])
st.write(hist)

balance_labels = False
if(balance_labels):
    st.markdown("### Before balancing")
    # Report number of positive examples
    positive_sent = webmddf[webmddf['sentiment']==1]
    st.write('There are {} positive reviews'.format(len(positive_sent)))

    # Report number of negative examples
    negative_sent = webmddf[webmddf['sentiment']==-1]
    st.write('There are {} negative reviews'.format(len(negative_sent)))

    st.markdown("### After balancing")
    # Sample number of negative example from positive examples (# positive > # negative)
    positive_sample = positive_sent.sample(n = len(negative_sent))
    st.write('[Update] There are {} positive reviews'.format(len(positive_sample)))

    # Merge positive and negative examples and update products dataframe
    frames = [negative_sent, positive_sample]
    webmddf = pd.concat(frames)

#splitting dataset into train and test sets
X, y = webmddf.loc[:, ~webmddf.columns.isin(['sentiment'])], webmddf.loc[:, webmddf.columns.isin(['sentiment'])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1)

def predict_probability(X, W, b=0):
    # Take dot product of feature_matrix and coefficients  
    score = np.dot(X, W)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    y_pred = 1. / (1.+np.exp(-score)) + b    
    return y_pred

def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)/len(feature_matrix)
    
    return lp

def update_weights(X, Y, W, b, learning_rate, log_likelihood):      
    # no_of_training_examples, no_of_features         
    num_features, num_examples = X.shape
    # Make a prediction
    y_pred = 1 / (1 + np.exp(-(X.dot(W) + b))) 
    
    dW = X.T.dot(Y-y_pred) / num_features 
    db = np.sum(Y-y_pred) / num_features 

    # update weights and bias
    b = b + learning_rate * db
    W = W + learning_rate * dW

    # Compute log-likelihood
    for i in range(len(W)):
        #y_pred = 1 / (1 + np.exp(-(X[:,i].dot(W[i]) + b))) 
        #log_likelihood += compute_avg_log_likelihood(X[:,i], Y, W[i])
        log_likelihood = compute_avg_log_likelihood(X[:,i], Y, W[i])

    return W, b, log_likelihood

def predict(X, W, b):
    Z = 1 / (1 + np.exp(- (X.dot(W) + b)))
    Y = [-1 if z <= 0.5 else +1 for z in Z]
    return Y

def fit(X, Y, num_iterations, learning_rate):   
    # no_of_training_examples, no_of_features         
    num_features, num_examples = X.shape    
    
    # weight initialization         
    W = np.zeros(num_examples)
    b = 0
    log_likelihood=0
    likelihood_history=[]
      
    # gradient ascent learning 
    for i in range(num_iterations):          
        W, b, log_likelihood = update_weights(X, Y, W, b, learning_rate, log_likelihood)   
        likelihood_history.append(log_likelihood)
    return W, b, likelihood_history

# Word count
feature_encoding = st.session_state['feature_encoding']
print(feature_encoding)
if(feature_encoding=='word count'):
    X_train_sentiment = X_train.loc[:,X_train.columns.str.startswith('word_count_')]
    X_test_sentiment = X_test.loc[:,X_test.columns.str.startswith('word_count_')]

# TF-IDF
if(feature_encoding=='tfidf word count'):
    X_train_sentiment = X_train.loc[:,X_train.columns.str.startswith('tfidf_word_count_')]
    X_test_sentiment = X_test.loc[:,X_test.columns.str.startswith('tfidf_word_count_')]

st.session_state['X_train_sentiment'] = X_train_sentiment
st.session_state['X_test_sentiment'] = X_test_sentiment

#X_train_sentiment.head()
# learning parameters
st.markdown("### Select learning parameters")
lg_col1, lg_col2 = st.columns(2)

with (lg_col1):
    learning_rate = float(st.text_input(
        label='Input learning rate 👇',
        value='0.0001',
        key='lg_learning_rate_textinput'
    ))
st.write('You selected the following learning rate: {}'.format(learning_rate))

with (lg_col2):
    # Maximum iterations to run the LG until convergence
    num_iterations = st.number_input(
        label='Enter the number of maximum iterations on training data',
        min_value=1,
        max_value=5000,
        value=10,
        step=100,
        key='lg_max_iter_numberinput'
    )
    st.write('You set the maximum iterations to: {}'.format(num_iterations))

st.session_state['learning_rate'] = learning_rate
st.session_state['num_iterations'] = num_iterations
print('started fitting')
sentiment_model_weights, sentiment_model_bias, likelihood_history = fit(X_train_sentiment.to_numpy(), np.ravel(y_train), num_iterations, learning_rate)
print('done fitting')
plt.scatter(np.arange(0,len(likelihood_history),1), likelihood_history, color = 'blue') 
plt.title('Log Likelihood vs Training Iteration') 
plt.xlabel('Training Iterations') 
plt.ylabel('Log Likelihood') 
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(plt)

review_idx=10
sentiment = predict(X_test_sentiment[:review_idx], sentiment_model_weights, sentiment_model_bias)
sentiment = ['positive' if i==1 else 'negative' for i in sentiment]
print(sentiment)

sentiment = predict(X_test_sentiment[:2], sentiment_model_weights, sentiment_model_bias)
print(sentiment)

# print(webmddf['Reviews'][:2])
# print(webmddf['sentiment'][:2])

# st.markdown("### Model weight")
# st.write(sentiment_model_weights)
# st.markdown("### Model bias")
# st.write(sentiment_model_bias)

num_positive_weights = np.sum(sentiment_model_weights >= 0)
num_negative_weights = np.sum(sentiment_model_weights < 0)

st.markdown("### Number of positive/negative weights")
st.write("Number of positive weights: %s " % num_positive_weights)
st.write("Number of negative weights: %s " % num_negative_weights)

# def get_classification_accuracy(prediction_labels, true_labels):    
#     # Compute the number of correctly classified examples
#     num_correct = np.sum(prediction_labels == true_labels)

#     # Then compute accuracy by dividing num_correct by total number of examples
#     accuracy = num_correct / len(true_labels)
#     return accuracy

# accuracy = get_classification_accuracy(predict(X_train_sentiment.to_numpy(), sentiment_model_weights, sentiment_model_bias), 
#                                        np.ravel(y_train))
# st.markdown("### Accuracy")
# st.write(accuracy)

def plot_series(data, x_title, y_title, para1, para2, legend_label='iteration '):
    colors = mcolors.TABLEAU_COLORS

    # Sort colors by hue, saturation, value and name.
    names = sorted(
        colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    # Plot cost of training iterations
    j=0    
    for i in range(para1,len(data),para2):
        plt.plot(np.arange(0,len(data[i]),1), data[i], color = names[j], label=legend_label+str(i))
        if(j>=len(names)-1):
            j=0
        else:
            j+=1
    plt.title(y_title+' vs '+x_title+' Iteration') 
    plt.xlabel(x_title) 
    plt.ylabel(y_title) 
    plt.legend()
    st.pyplot(plt.show())

####### Commented this out because of time #######
weights_list=[]
likelihood_history = []
for lr in range(1,50,1):
    learning_rate=lr/100000
    weights, bias, log_lik = fit(X_train_sentiment.to_numpy(), np.ravel(y_train), num_iterations, learning_rate)
    likelihood_history.append(log_lik)
st.markdown("### Training iteration")
if (st.button('Train iterations Model')):
    plot_series(likelihood_history, 'Training Iteration', 'Log Likelihood', 0, 10, legend_label='lr-')

st.session_state['X_train'] = X_train
st.session_state['X_test'] = X_test
st.session_state['y_train'] = y_train
st.session_state['y_test'] = y_test

st.markdown("Click **Test Model** to test the model.")