import pandas as pd
import numpy as np

df = pd.read_csv('./diabetes.csv')
X = df.drop(['Outcome'], axis=1)
y = df[['Outcome']]
split = int(X.shape[0]*0.8)
trian, test = df.iloc[:split], df.iloc[split:]

def calculate_prior(df, Y): ## P(Y=y) for all y in classes
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y] == i])/len(df))
    return prior

def calculate_likelihood_gaussian(df, feature_name, feature_val, Y, class_label): ## P(X=x | Y=y)
    features = list(df.columns)
    df = df[df[Y] == class_label] ## select 
    mean, std = df[feature_name].mean(), df[feature_name].std()
    ## P(X|Y) = 1/(sqrt(2pi)*std) * exp( -(x-mean)^2 / 2*variance)
    p_x_given_y = (1 /
                 (np.sqrt(2 * np.pi) * std)) * np.exp(-((feature_val - mean)**2 /
                                                        (2 * std**2)))
    return p_x_given_y

def naive_bayes_classifier(df, X, Y):
    features = list(df.columns)[:-1]
    prior = calculate_prior(df, Y)
    labels = sorted(list(df[Y].unique()))
    Y_pred = []
    for x in X:
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        post_prob = [1]* len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j]*prior[j]
        Y_pred.append(np.argmax(post_prob))
    return np.array(Y_pred)

## Given a df, and X_test values, calculate P(Y|X) post_prob = P(X|Y) likelihood * P(Y=y) prior prob
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]
y_pred = naive_bayes_classifier(trian, X_test.to_numpy(), Y='Outcome')
y_test = y_test.to_numpy()

accuracy = np.sum( y_pred == y_test)/len(y_pred)

print("Accuracy = ", accuracy)