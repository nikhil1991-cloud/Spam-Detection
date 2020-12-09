import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Read the csv file
df = pd.read_csv('/Users/nikhil/Data/ML_examples/spam.csv',encoding = 'latin-1')
#Replace spam/ham with 1/0
df['v1'] = df['v1'].replace(['ham','spam'],[0,1])
#Rename columns
df = df.rename(columns={'v1': 'label'})
df = df.rename(columns={'v2': 'message'})
df = df[['label','message']]
variables = df.columns
#Perform train test split
df_shuffle = df.sample(frac=1)
train_split = int(1*len(df_shuffle))#Since we will use our own spam/ham test input, the whole data set of spam.csv is the trained set
df_train = np.array(df_shuffle[:train_split])
#see the word cloud for spam emails
spam_words =' '.join(list(df['message'][df['label']==1]))
spam_wc = WordCloud(width=500,height=500).generate(spam_words)
#see the word cloud for ham email
ham_words =' '.join(list(df['message'][df['label']==0]))
ham_wc = WordCloud(width=500,height=500).generate(ham_words)
#Uncomment lines 30 31 to plot wordcloud
plt.imshow(ham_wc)
plt.axis('off')

#Data processiong
def processing(message):
    message=message.lower() #lowercase all words
    words = word_tokenize(message) #separate them from punctuation
    words = [w for w in words if len(w) > 2] #select on those words with length > 2
    sw = stopwords.words('english') #Removing stop words
    words = [k for k in words if k not in sw]
    stemmer = PorterStemmer()#Stremming
    words = [stemmer.stem(k) for k in words]
    return words


#Aim to calculate P(spam|w) given a word w using Naive Bayes
#P(spam|w) = P(w|spam)*P(spam)
#P(w|spam) = [TF(w|spam) + alpha]/[TF(x) + alpha*N_vocab] ; x is all the words in the train set; TF is term frequency; alpha is additive smoothing
#simiarly for P(ham|w)
#so given an input word classify spam or ham by argmax[P(spam|w),P(ham|w)]

#calulate P(spam) and P(ham)
num_spam = len(np.where(df_train[:,0]==1)[0])
total = df_train.shape[0]
p_spam = num_spam/total
p_not_spam = (total - num_spam)/total
#Calculate N_vocab(list of unique words)
N_vocab_all = []
i=0
for i in range (0,len(df_train)):
    w_all = processing(df_train[i,1])
    j =0
    for j in range (0,len(w_all)):
        N_vocab_all.append(w_all[j])
N_vocab = len(np.unique(N_vocab_all))

#Type in any message which you believe is a spam or a ham
stop='no'
while stop != 'yes':
 Input = str(input("Enter your message: "))
 specific_message = Input
 target_words = processing(specific_message)
 alpha=1
 w=0
 TF_ws = np.zeros(np.shape(target_words))
 TF_xs = np.zeros(np.shape(target_words))
 TF_xh = np.zeros(np.shape(target_words))
 TF_wh = np.zeros(np.shape(target_words))
 for w in range (0,len(target_words)):
    wh_freq,ws_freq,xs_freq,xh_freq = 0,0,0,0
    i=0
    for i in range (0,len(df_train)):
        train_words = np.array(processing(df_train[i,1]))
        train_label = df_train[i,0]
        if (train_label==1):
           ws_freq += len(np.where(target_words[w] == train_words)[0])
           xs_freq += len(train_words)
        if (train_label==0):
           wh_freq += len(np.where(target_words[w] == train_words)[0])
           xh_freq += len(train_words)
    TF_ws[w] = ws_freq
    TF_xs[w] = xs_freq
    TF_xh[w] = xh_freq
    TF_wh[w] = wh_freq
    

 Num_spam = TF_ws + alpha
 Num_ham = TF_wh + alpha
 Den_spam = TF_xs + alpha*N_vocab
 Den_ham = TF_xh + alpha*N_vocab

 Prob_spam = np.prod(Num_spam/Den_spam)*p_spam
 Prob_ham = np.prod(Num_ham/Den_ham)*p_not_spam
 print('\n')
 print('\n')
 print('The message you got is: '+ specific_message)
 print('\n')
 if Prob_ham>Prob_spam:
   print('This is not a spam!')
 else:
   print('This is a spam!')
 print('\n')
 stop = str(input("Are you done? Enter 'yes' to stop / 'no' to continue:"))
 if stop=='yes':
    print('Thank you!')
