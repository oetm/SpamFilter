from sklearn.feature_extraction.text import CountVectorizer

train_text = [
    'This is the my name name name first phrase is',
    'This is the second one the the',
    'And this the last phrase last the and and'
]

test_text = ['My name name name anme is Teo']

count_vect = CountVectorizer(lowercase = True, analyzer = 'word')
count_vect.fit(train_text)

X = count_vect.transform(train_text)

print(X.toarray())
print(count_vect.get_feature_names())

X_test = count_vect.transform(test_text)
print(X_test.toarray())