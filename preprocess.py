import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv('data/earthquake_alert_balanced_dataset.csv')
scaler = StandardScaler()
numerical_cols = ['magnitude', 'depth', 'cdi','mmi','sig']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['alert']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['alert']))
df = pd.concat([df, encoded_df], axis=1)
df.drop('alert', axis=1, inplace=True)
train=df.iloc[:,:1200]
train.to_csv('data/train_data.csv',index=False)
test=df.iloc[:,1200:]
test.to_csv('data/test_data.csv',index=False)
