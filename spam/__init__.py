from numpy import loadtxt
import pandas

df = pandas.read_table('SMSSpamCollection.txt', sep='\t', header=None, names=['label', 'sms_message'])

df['label'] = df.label.map({'ham':0, 'spam':1})

print(df.shape)

print df.head()
