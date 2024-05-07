from pages.Train_Model import predict_probability, predict, fit, get_classification_accuracy, train_test_split
import streamlit as st                  
import numpy as np    
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt

num_iterations = st.session_state['num_iterations']
learning_rate = st.session_state['learning_rate']

X_train_sentiment = st.session_state['X_train_sentiment']
X_test_sentiment = st.session_state['X_test_sentiment']

X_train = st.session_state['X_train']
X_test = st.session_state['X_test']
y_train = st.session_state['y_train']
y_test = st.session_state['y_test']
webmddf = st.session_state['data']

st.markdown("### View data")
st.write(webmddf.head())

sentiment_model_weights, sentiment_model_bias, likelihood_history = fit(X_train_sentiment.to_numpy(), np.ravel(y_train), num_iterations, learning_rate)

review_idx=10
sentiment = predict(X_test_sentiment[:review_idx], sentiment_model_weights, sentiment_model_bias)
sentiment = ['positive' if i==1 else 'negative' for i in sentiment]

sentiment = predict(X_test_sentiment[:2], sentiment_model_weights, sentiment_model_bias)

# print(webmddf['Reviews'][:2])
# print(webmddf['sentiment'][:2])

# st.markdown("### Model weight")
# st.write(sentiment_model_weights)
# st.markdown("### Model bias")
# st.write(sentiment_model_bias)

# most positive or negative review

# Calculate the class probabilities for the test set
y_prob = predict_probability(X_test_sentiment.to_numpy(), sentiment_model_weights, b=sentiment_model_bias)

# Sort the test set in descending order of their probabilities of being positive
idx = np.argsort(-y_prob)

# Get the indices of the 20 most positive reviews
idx_most_positive = idx[-20:] 
print(idx_most_positive)

# Get the corresponding reviews from the test set
most_positive_reviews = X_test.iloc[idx_most_positive]
st.markdown("### Most positive reviews")
st.write(most_positive_reviews)

# Get the indices of the 20 most negative reviews
idx_most_negative = idx[:20]

# Get the corresponding reviews from the test set
most_negative_reviews = X_test.iloc[idx_most_negative]
st.markdown("### Most negative reviews")
st.write(most_negative_reviews)


accuracy = get_classification_accuracy(predict(X_test_sentiment.to_numpy(), sentiment_model_weights, sentiment_model_bias), 
                                       np.ravel(y_test))
st.markdown("### Accuracy")
st.write(accuracy)

num_positive = int(np.sum(y_train == +1))
num_negative = int(np.sum(y_train == -1))
print(num_positive)
print(num_negative)

if num_positive >= num_negative:
    y_pred = 1
else:
    y_pred = -1

n_correct = np.sum(y_test == y_pred)
accuracy = n_correct / len(X_test)
print(accuracy)

#X, y = webmddf.loc[:, ~webmddf.columns.isin(['sentiment'])], webmddf.loc[:, webmddf.columns.isin(['sentiment'])]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1)

num_iterations=100

sentiment_model = LogisticRegression(random_state=0, max_iter=num_iterations, tol=1e-3)

#fit(X_train_sentiment.to_numpy(), np.ravel(y_train), num_iterations, learning_rate)
sentiment_model.fit(X_train_sentiment.to_numpy(), np.ravel(y_train))

#(X_test_sentiment[:review_idx], sentiment_model_weights, sentiment_model_bias)
feature_encoding = st.session_state['feature_encoding']
if feature_encoding == "word count":
    sentiment_predictions = sentiment_model.predict(X_test_sentiment.loc[:, X_test_sentiment.columns.str.startswith('word_count_')])
if feature_encoding == "tfidf word count":
    sentiment_predictions = sentiment_model.predict(X_test_sentiment.loc[:, X_test_sentiment.columns.str.startswith('tfidf_word_count_')])
cmatrix = confusion_matrix(y_test, sentiment_predictions)

st.markdown("### False/True positives and negatives")

true_neg, false_pos, false_neg, true_pos = cmatrix.ravel()
st.write('There are {} false positives'.format(false_pos))
st.write('There are {} false negatives'.format(false_neg))
st.write('There are {} true positives'.format(true_pos))
st.write('There are {} true negatives'.format(true_neg))

st.markdown("### Precision and Recall")

precision = true_pos/(true_pos+false_pos)
st.write("Precision on test data: %s" % precision)

#false_pos / (true_pos + false_pos)

recall = true_pos / (true_pos + false_neg)
st.write("Recall on test data: %s" % recall)

#true_pos / (true_pos + false_neg)

# Varying the threshold
def apply_threshold(probabilities, threshold):
    # +1 if >= threshold and -1 otherwise.
    return np.array([1 if p >= threshold else -1 for p in probabilities])

probabilities = predict_probability(X_test_sentiment.to_numpy(), sentiment_model_weights)

predictions_with_default_threshold = apply_threshold(probabilities, 0.5)

predictions_with_high_threshold = apply_threshold(probabilities, 0.9)

# Threshold = 0.5
precision_with_default_threshold = precision_score(
    y_test, predictions_with_default_threshold)

recall_with_default_threshold = recall_score(
    y_test, predictions_with_default_threshold
)

# Threshold = 0.9
precision_with_high_threshold = precision_score(
    y_test, predictions_with_high_threshold, zero_division=1)

recall_with_high_threshold = recall_score(
    y_test, predictions_with_high_threshold
)

st.markdown("### Precision/Recall as threshold varies")

st.write("Precision (threshold = 0.5): %s" % precision_with_default_threshold)
st.write("Recall (threshold = 0.5)   : %s" % recall_with_default_threshold)

st.write("Precision (threshold = 0.9): %s" % precision_with_high_threshold)
st.write("Recall (threshold = 0.9)   : %s" % recall_with_high_threshold)


# precision and recall curve
threshold_values = np.linspace(0.5, 1, num=100)

precision_all = []
recall_all = []
# For each of the values of threshold, we compute the precision and recall scores.
probabilities = predict_probability(X_test_sentiment.to_numpy(), sentiment_model_weights)
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)

    precision = precision_score(y_test, predictions, zero_division=1)

    recall = recall_score(y_test, predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(recall, precision, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.rcParams.update({'font.size': 16})

st.markdown("### Precision recall curve")
st.pyplot(plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)'))

for idx, precision in enumerate(precision_all):
    if precision >= 0.965:
        st.write("The smallest threshold value that achieves a precision of 96.5'%' or better: " + str(threshold_values[idx]))
        break


#Using `threshold` = 0.98, how many **false negatives** do we get on the **test_data**?
st.markdown("### Test the impact of threshold on false negatives in test data")
threshold = float(st.text_input(label='Input threshold 👇',
        value='0.949',
        key='threshold_textinput'))
predictions = apply_threshold(probabilities, threshold)
cm = confusion_matrix(y_test, predictions)
st.write("We get " + str(cm[1][0]) + " false negatives in test data")