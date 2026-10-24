import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def feature_engineering(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)')[0]
    df['Age'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior', 'Elderly'])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.fillna({'Age': df['Age'].mode()[0], 'Embarked': 'S'}, inplace=True)
    return df

pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'FamilySize']),
            ('cat', OneHotEncoder(), ['Title', 'Embarked'])
        ]
    ))
])

# Sample for testing
if __name__ == '__main__':
    df = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
    df = feature_engineering(df)
    X = pipeline.fit_transform(df)
    print(X)
