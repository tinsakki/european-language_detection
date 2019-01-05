from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
labels = {
'bg': "Bulgarian",
'cs': "Czech",
'da': "Danish",
'de': "German",
'el': "Greek, Modern",
'en': "English",
'es': "Spanish",
'et': "Estonian",
'fi': "Finnish",
'fr': "French",
'hu': "Hungarian",
'it': "Italian",
'lt': "Lithuanian",
'lv': "Latvian",
'nl': "Dutch",
'pl': "Polish",
'pt': "Portuguese",
'ro': "Romanian",
'sk': "Slovak",
'sl': "Slovenian",
'sv': "Swedish",
}
languages = pd.read_excel('language.xlsx')
df_languages = pd.DataFrame(languages)
df_languages.columns = ['language code','natural language', ]
language_features = df_languages['natural language']
language_targets = df_languages['language code']
X_train, X_test, y_train, y_test = train_test_split(language_features,
                                                    language_targets,
                                                    test_size = 0.3,
                                                    random_state = 42)
tfidf_vect = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
X_train = tfidf_vect.fit_transform(X_train.values.astype('U'))
X_test = tfidf_vect.transform(X_test.values.astype('U'))
model = MultinomialNB()
model.fit(X_train, y_train)
with open('model.pkl', 'wb') as file:
    pickle.dump((tfidf_vect, model), file)
def load():
    with open('model.pkl', 'rb') as file:
      vectorizer, clf = pickle.load(file)
    return vectorizer, clf

predictions = model.predict(X_test)
ans = accuracy_score(y_test,predictions)
print(ans)
vectorizer, classifer = load()
email_input = ["Who knows ?"]
email_input_transformed = vectorizer.transform(email_input)
prediction = classifer.predict(email_input_transformed)
for i,j in labels.items():
    if i==prediction:
        print(j)