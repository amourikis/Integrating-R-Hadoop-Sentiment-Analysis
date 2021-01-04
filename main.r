# Sentiment Analysis with "sentiment"
# 
# R package "sentiment"
# 
#   classify_emotion
# This function helps us to analyze some text and classify it in different types of emotion: anger, disgust, fear, joy, sadness, and surprise. The classification can be performed using two algorithms: one is a naive Bayes classiï¬er trained on Carlo Strapparava and Alessandro Valituttiâs emotions lexicon; the other one is just a simple voter procedure.
# 
# classify_polarity
# In contrast to the classification of emotions, the classify_polarity function allows us to classify some text as positive or negative. In this case, the classification can be done by using a naive Bayes algorithm trained on Janyce Wiebeâs subjectivity lexicon; or by a simple voter algorithm.
# 
# Important Note:
#   The R package "sentiment" depends on Duncan's Temple Rstem package that is only available at Omegahat
# At the time of this writing, I'm using the version 0.4-1 (I downloaded and installed the tar.gz file from the package website).
# 
# required pakacges
library(plyr)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(devtools)
install_github('sentiment140', 'okugami79')
library("devtools")
install_github("sentiment", "andrie")
library(tm)  
library(Rstem)
install.packages("Rstem")
library("devtools")
install_github("andrie/sentiment", force = TRUE)
library(sentiment)
load_all('E:/shivam_freelancer/fred/code_n_plots/code/sentiment')
library(sentiment)
library(plyr)
require(Rcpp)

library(RSentiment)

Sys.setenv(JAVA_HOME='D://Program Files/Java/jre1.8.0_112') # for 64-bit version
library(rJava)



#loading libraries
library('stringr')
library('readr')
library('wordcloud')
library('tm')
library('SnowballC')
library('RSentiment')
library('DT')



tweets_df <- read.csv("E://shivam_freelancer/fred/data/final_extracted_data.csv")
View(tweets_df)

tweets <- tweets_df$text
# remove retweet entities
tweets = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", tweets)
# remove at people
tweets = gsub("@\\w+", "", tweets)
# remove punctuation
tweets = gsub("[[:punct:]]", "", tweets)
# remove numbers
tweets = gsub("[[:digit:]]", "", tweets)
# remove html links
tweets = gsub("http\\w+", "", tweets)
# remove unnecessary spaces
tweets = gsub("[ \t]{2,}", "", tweets)
tweets = gsub("^\\s+|\\s+$", "", tweets)

# define "tolower error handling" function 
try.error = function(x)
{
  # create missing value
  y = NA
  # tryCatch error
  try_error = tryCatch(tolower(x), error=function(e) e)
  # if not an error
  if (!inherits(try_error, "error"))
  y = tolower(x)
  # result
  return(y)
}
# lower case using try.error with sapply 
tweets = sapply(tweets, try.error)

# remove NAs in tweets
tweets = tweets[!is.na(tweets)]
names(tweets) = NULL




# classify emotion
class_emo = classify_emotion(tweets)
# get emotion best fit
emotion = class_emo[,7]
# substitute NA's by "unknown"
emotion[is.na(emotion)] = "neutral"

# classify polarity
class_pol = classify_polarity(tweets, algorithm = "bayes")
# get polarity best fit
polarity = class_pol[,4]


# data frame with results
sent_df = data.frame(text = tweets, emotion = emotion,
                     polarity = polarity, stringsAsFactors = FALSE)

View(sent_df)

# sort data frame
sent_df = within(sent_df,
                 emotion <- factor(emotion, levels=names(sort(table(emotion), decreasing=TRUE))))


# plot distribution of emotions
ggplot(sent_df, aes(x=emotion)) +
geom_bar(aes(y=..count.., fill=emotion)) +
scale_fill_brewer(palette="Dark2") +
labs(x="emotion categories", y="number of tweets")

dev.off()

# plot distribution of polarity
ggplot(sent_df, aes(x=polarity)) +
geom_bar(aes(y=..count.., fill=polarity)) +
scale_fill_brewer(palette="RdGy") +
labs(x="polarity categories", y="number of tweets")


# separating text by emotion
emos = levels(factor(sent_df$emotion))
nemo = length(emos)
emo.docs = rep("", nemo)
for (i in 1:nemo)
{
  tmp = tweets[emotion == emos[i]]
  emo.docs[i] = paste(tmp, collapse =" ")
}

# remove stopwords
emo.docs = removeWords(emo.docs, stopwords("english"))
# create corpus
corpus = Corpus(VectorSource(emo.docs))
tdm = TermDocumentMatrix(corpus)
tdm = as.matrix(tdm)
colnames(tdm) = emos

# comparison word cloud
comparison.cloud(tdm, colors = brewer.pal(nemo, "Dark2"),
scale = c(3,.5), random.order = FALSE, title.size = 1.5)


ggplot(data = sent_df, aes(sent_df$polarity)) + 
  geom_histogram(stat="count") + 
  geom_density(col = 2) + 
  labs(title="Histogram for Polarity of Tweeets on Trump") +
  labs(x = "Tweets on Donald Trump", y = "Polarity")



df <- cbind(tweets_df, sent_df)
View(df)
keeps <- c( "text", "Favorites", "Retweets",  "Tweet.ID", "emotion", "polarity")
tweets_df <- df[keeps]
View(tweets_df)
tweets_df

df$len_words <- nchar(as.character(df$text))
View(df)

x = df$text
df$words_per_sentence <- sapply(gregexpr("\\S+", x), length)
write.csv(df, "E://shivam_freelancer/fred/data/new_extracted_data.csv", row.names=FALSE)





#########################################################################################

# sentiment analysis
score.sentiment = function(sentence, pos.words, neg.words)
{
  require(plyr)
  require(stringr)
  
  # clean up sentences with R's regex-driven global substitute, gsub():
  sentence = gsub('[[:punct:]]', '', sentence)
  sentence = gsub('[[:cntrl:]]', '', sentence)
  sentence = gsub('\\d+', '', sentence)
  # and convert to lower case:
  sentence = tolower(sentence)
  
  # split into words. str_split is in the stringr package
  word.list = str_split(sentence, '\\s+')
  # sometimes a list() is one level of hierarchy too much
  words = unlist(word.list)
  
  # compare our words to the dictionaries of positive & negative terms
  pos.matches = match(words, pos.words)
  neg.matches = match(words, neg.words)
  
  # match() returns the position of the matched term or NA
  # we just want a TRUE/FALSE:
  pos.matches = !is.na(pos.matches)
  neg.matches = !is.na(neg.matches)
  
  # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
  score = sum(pos.matches) - sum(neg.matches)
  
  return(score)
}


tweets_df <- df
colnames(tweets_df)
#extracting relevant data
r1 = as.character(tweets_df$text)


# Creating Corpus of Documents (ot tweets) and cleaning the tweets
set.seed(100)
sample = sample(r1, (length(r1)))
corpus = Corpus(VectorSource(list(sample)))
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, removeWords, stopwords('english'))
corpus = tm_map(corpus, stemDocument)
dtm_up = DocumentTermMatrix(VCorpus(VectorSource(corpus[[1]]$content)))
freq_up <- colSums(as.matrix(dtm_up))




#Calculating Sentiments
sentiments_up = calculate_sentiment(names(freq_up))
sentiments_up = cbind(sentiments_up, as.data.frame(freq_up))
sent_pos_up = sentiments_up[sentiments_up$sentiment == 'Positive',]
sent_neg_up = sentiments_up[sentiments_up$sentiment == 'Negative',]


#Positive Words
DT::datatable(sent_pos_up)
layout(matrix(c(1, 2), nrow=2), heights=c(1, 4))
par(mar=rep(0, 4))
plot.new()
set.seed(100)
wordcloud(sent_pos_up$text,sent_pos_up$freq,min.freq=50,colors=brewer.pal(100,"Dark2"))


#Negative Sentiments
DT::datatable(sent_neg_up)
#Word Cloud of Negative Words
plot.new()
set.seed(100)
wordcloud(sent_neg_up$text,sent_neg_up$freq, min.freq=50,colors=brewer.pal(100,"Dark2"))





#visual
library(ggplot2) # Data visualization
library(syuzhet)

#Approach 2 - using the 'syuzhet' package
text = as.character(tweets_df$text) 
##removing Retweets
some_txt<-gsub("(RT|via)((?:\\b\\w*@\\w+)+)","",text)
##let's clean html links
some_txt<-gsub("http[^[:blank:]]+","",some_txt)
##let's remove people names
some_txt<-gsub("@\\w+","",some_txt)
##let's remove punctuations
some_txt<-gsub("[[:punct:]]"," ",some_txt)
##let's remove number (alphanumeric)
some_txt<-gsub("[^[:alnum:]]"," ",some_txt)

mysentiment<-get_nrc_sentiment((some_txt))


# Get the sentiment score for each emotion
mysentiment.positive =sum(mysentiment$positive)
mysentiment.anger =sum(mysentiment$anger)
mysentiment.anticipation =sum(mysentiment$anticipation)
mysentiment.disgust =sum(mysentiment$disgust)
mysentiment.fear =sum(mysentiment$fear)
mysentiment.joy =sum(mysentiment$joy)
mysentiment.sadness =sum(mysentiment$sadness)
mysentiment.surprise =sum(mysentiment$surprise)
mysentiment.trust =sum(mysentiment$trust)
mysentiment.negative =sum(mysentiment$negative)


# Create the bar chart
yAxis <- c(mysentiment.positive,
           + mysentiment.anger,
           + mysentiment.anticipation,
           + mysentiment.disgust,
           + mysentiment.fear,
           + mysentiment.joy,
           + mysentiment.sadness,
           + mysentiment.surprise,
           + mysentiment.trust,
           + mysentiment.negative)


xAxis <- c("Positive","Anger","Anticipation","Disgust","Fear","Joy","Sadness","Surprise","Trust","Negative")
colors <- c("green","red","blue","orange","red","green","orange","blue","green","red")
yRange <- range(0,yAxis) + 1000
barplot(yAxis, names.arg = xAxis, 
        xlab = "Emotion", ylab = "Score", main = "Twitter sentiment for Donald Trump's Tweets", sub = "Feb 2017", col = colors, border = "black", ylim = yRange, xpd = F, axisnames = T, cex.axis = 0.8, cex.sub = 0.8, col.sub = "blue")
colSums(mysentiment)







##################################################################################################################
## SOME OTHER VISUALIZATIONS  ####################################################################################
##################################################################################################################

## Some other visualizations
library(dplyr)


df <- read.csv("E://shivam_freelancer/fred/data/new_extracted_data.csv")

keeps <- c( "text", "Favorites", "Retweets",  "Tweet.ID", "emotion", "polarity",  "len_words","words_per_sentence")
tweets_df <- df[keeps]
df1 <- tweets_df


# Scatterplot
gg <- ggplot(df1, aes(x=text, y=Favorites)) + 
  geom_point(aes(col=text, size=Favorites)) + 
  geom_smooth(method="loess", se=F)
  labs(y="Favts", 
       x="text", 
       title="Tweets versus Favorites")

plot(gg)

library(ggplot2)
theme_set(theme_bw())

# Draw plot
ggplot(df1, aes(x=text, y=Favorites)) + 
  geom_bar(stat="identity", width=.5, aes(color = polarity)) + 
  labs(title="Total Number of Tweets and Favourite Counts")

## Retweets
# Set a unique color with fill, colour, and alpha
# Scatterplot
gg <- ggplot(df1, aes(x=text, y=Retweets)) + 
  geom_point(aes(col=polarity, size=Retweets)) + 
  geom_smooth(method="loess", se=F)

plot(gg)

library(ggplot2)
theme_set(theme_bw())

# Draw plot
ggplot(df1, aes(x=text, y=Retweets)) + 
  geom_bar(stat="identity", width=.5, aes(color = polarity)) + 
  labs(title="Total Number of Tweets and Retweets")


g <- ggplot(df1, aes(len_words)) + scale_fill_brewer(palette = "Set3")
g + geom_histogram(aes(fill=polarity), 
                   binwidth = 1, 
                   col="grey", 
                   size=1) +  # change binwidth
  labs(title="Number of Words in the tweets")  


# words_per_sentence
g <- ggplot(df1, aes(words_per_sentence)) + scale_fill_brewer(palette = "Spectral")
g + geom_histogram(aes(fill=polarity), 
                   binwidth = 1, 
                   col="grey", 
                   size=1) +  # change binwidth
  labs(title="Number of Words in the tweets")  




g <- ggplot(df1, aes(x="", y=text, fill=polarity))+
  geom_bar(width = 1, stat = "identity")

pie <- g + coord_polar("y", start=0)

pie + scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))






pie + scale_fill_brewer("Blues") + 
  theme(axis.text.x=element_blank())+
  geom_text(aes(y = tweet + c(0, cumsum(tweet)[-length(tweet)]), 
                label = percent(tweet/100)), size=5)




























# Load required libraries

library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)

set.seed(1)
# Convert the 'class' variable from character to factor.
df$polarity <- as.factor(df$polarity)


# Bag of Words Tokenisation
corpus <- Corpus(VectorSource(df$text))
inspect(corpus[1:3])

#Data Cleanup
# Use dplyr's  %>% (pipe) utility to do this neatly.
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)


#Matrix representation of Bag of Words : The Document Term Matrix
dtm <- DocumentTermMatrix(corpus)
# Inspect the dtm
inspect(dtm[40:50, 10:15])

#Partitioning the Data

n <- nrow(df)
shuffled <- df[sample(n),]
train <- shuffled[1:round(0.7 * n),]
test <- shuffled[(round(0.7 * n) + 1):n,]


n <- nrow(dtm)
shuffled <- dtm[sample(n),]
dtm.train <- shuffled[1:round(0.7 * n),]
dtm.test <- shuffled[(round(0.7 * n) + 1):n,]


corpus.clean.train <- shuffled[1:round(0.7 * n),]
corpus.clean.test <- shuffled[(round(0.7 * n) + 1):n,]



# Feature Selection

fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))

# Use only 5 most frequent words (fivefreq) to build the DTM
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dim(dtm.train.nb)



# Train the classifier
system.time( classifier <- naiveBayes(corpus.clean.train, train$polarity, laplace = 1) )

# Use the NB classifier we built to make predictions on the test set.
system.time( nbpred <- predict(classifier, newdata=corpus.clean.test) )

# Create a truth table by tabulating the predicted class labels with the actual class labels 
table("Predictions"= nbpred,  "Actual" = test$polarity )

# Prepare the confusion matrix
conf.mat <- confusionMatrix(pred, df.test$class)

conf.mat

conf.mat$byClass

conf.mat$overall

# Prediction Accuracy
conf.mat$overall['Accuracy']


######################################################
# SVM
######################################################
dtMatrix <- create_matrix(df["text"])

container <- create_container(dtMatrix, df$polarity, trainSize=1:length(df)*0.75, virgin=FALSE)

model <- train_model(container, "SVM", kernel = "linear", cost = 1)

predictionData <- sample(df$text, length(df)*0.75)
predMatrix <- create_matrix(predictionData, originalMatrix=dtMatrix)

#prediction
predSize = length(predictionData);
predictionContainer <- create_container(predMatrix, labels=rep(0,predSize), testSize=1:predSize, virgin=FALSE)

#classification
results <- classify_model(predictionContainer, model)
results

#confusion matrix
df.test <- df[length(df)*0.75,length(df)]
confsvm.mat <- confusionMatrix(results$SVM_LABEL,df.test$polarity)
confsvm.mat$overall['Accuracy']

#plots frequency of predictions
Overall <- results$SVM_LABEL
p<-ggplot(results, aes(Overall)) + geom_bar()
p




##################################################################################

library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)

df$text <- iconv(df$text,"WINDOWS-1252","UTF-8")

corpus <- Corpus(VectorSource(df$text))
inspect(corpus[1:3])

library(stringr)
df$text <- tolower(str_trim(df$text))

#DOCUMENT TERM MATRIX
dtm <- DocumentTermMatrix(corpus.clean)


inspect(dtm[40:50, 10:15])

#FEATURE SELECTION
frequency <- findFreqTerms(dtm, 7)
dtm <- DocumentTermMatrix(corpus.clean, control=list(dictionary = frequency))

#Convert word frequencies to yes and no
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("Negative", "Positive"))
  y
}

dtm <- apply(dtm, 2, convert_count)

dtm<-as.data.frame(as.matrix(dtm))

#NAIVE BAYES
naive.model <- naiveBayes(dtm, df$rating, laplace = 1)
pred <- predict(naive.model, newdata=dtm.test)

#Prediction table
table("Predictions"= pred,  "Actual" = df.test$rating)

#Confusion matrix
conf.mat <- confusionMatrix(pred, df.test$rating)
conf.mat$overall['Accuracy'] * 100



#########################################################
# Load libs
library(readr)	# read in big data a bit quicker
library(tm)	# pre_process text
library(glmnet)	# for lasso
library(SnowballC) #for stemming
library(kernlab)# for svm
library(dplyr)

set.seed(1)
# Convert the 'class' variable from character to factor.
df$polarity <- as.factor(df$polarity)

df <- as.data.frame(df)
colnames(df) 	<- make.names(colnames(df))

n <- nrow(df)
shuffled <- df[sample(n),]
train <- shuffled[1:round(0.7 * n),]
test <- shuffled[(round(0.7 * n) + 1):n,]

# An abstract function to preprocess a text column
preprocess <- function(text_column)
{
  # Use tm to get a doc matrix
  corpus <- Corpus(VectorSource(text_column))
  # all lower case
  corpus <- tm_map(corpus, content_transformer(tolower))
  # remove punctuation
  corpus <- tm_map(corpus, content_transformer(removePunctuation))
  # remove numbers
  corpus <- tm_map(corpus, content_transformer(removeNumbers))
  # remove stopwords
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  # stem document
  corpus <- tm_map(corpus, stemDocument)
  # strip white spaces (always at the end)
  corpus <- tm_map(corpus, stripWhitespace)
  # return
  corpus	
}

# Get preprocess training and test data
train_corpus <- preprocess(train$text1)
test_corpus  <- preprocess(test$text1)

# Create a Document Term Matrix for train and test
# Just including bi and tri-grams

Sys.setenv(JAVA_HOME='D://Program Files/Java/jre1.8.0_112') # for 32-bit version
library(rJava)
library(RWeka)


# Bi-Trigram tokenizer function (you can always get longer n-grams)
bitrigramtokeniser <- function(x, n) {
  RWeka:::NGramTokenizer(x, RWeka:::Weka_control(min = 2, max = 3))
}

# Appropriate libraries 
# For Windows
# Sys.setenv(JAVA_HOME='C:/Program Files/Java/jre7/')
library(rJava)
library(RWeka)

"
Remove remove words <=2
TdIdf weighting
Infrequent (< than 1% of documents) and very frequent (> 80% of documents) terms not included
"
train_dtm <- DocumentTermMatrix(train_corpus, control=list(wordLengths=c(2, Inf), 
                                                           tokenize = bitrigramtokeniser, 
                                                           weighting = function(x) weightTfIdf(x, normalize = FALSE),
                                                           bounds=list(global=c(floor(length(train_corpus)*0.01), floor(length(train_corpus)*.8)))))

test_dtm <- DocumentTermMatrix(test_corpus, control=list(wordLengths=c(2, Inf), 
                                                         tokenize = bitrigramtokeniser, 
                                                         weighting = function(x) weightTfIdf(x, normalize = FALSE),
                                                         bounds=list(global=c(floor(length(test_corpus)*0.001), floor(length(test_corpus)*.8)))))

# Variable selection
# ~~~~~~~~~~~~~~~~~~~~
"
For dimension reduction.
The function calculates chi-square value for each phrase and keeps phrases with highest chi_square values
Ideally you want to put variable selection as part of cross-validation.

chisqTwo function takes:
document term matrix (dtm), 
vector of labels (labels), and 
number of n-grams you want to keep (n_out)

"
chisqTwo <- function(dtm, labels, n_out=2000){
  mat 		<- as.matrix(dtm)
  cat1		<- 	colSums(mat[labels==T,])	  	# total number of times phrase used in cat1 
  cat2		<- 	colSums(mat[labels==F,])	 	# total number of times phrase used in cat2 
  n_cat1		<- 	sum(mat[labels==T,]) - cat1   	# total number of phrases in soft minus cat1
  n_cat2		<- 	sum(mat[labels==F,]) - cat2   	# total number of phrases in hard minus cat2
  
  num 		<- (cat1*n_cat2 - cat2*n_cat1)^2
  den 		<- (cat1 + cat2)*(cat1 + n_cat1)*(cat2 + n_cat2)*(n_cat1 + n_cat2)
  chisq 		<- num/den
  
  chi_order	<- chisq[order(chisq)][1:n_out]   
  mat 		<- mat[, colnames(mat) %in% names(chi_order)]
  
}

"
With high dimensional data, test matrix may not have all the phrases training matrix has.
This function fixes that - so that test matrix has the same columns as training.
testmat takes column names of training matrix (train_mat_cols), and 
test matrix (test_mat)
and outputs test_matrix with the same columns as training matrix
"
# Test matrix maker
testmat <- function(train_mat_cols, test_mat){	
  # train_mat_cols <- colnames(train_mat); test_mat <- as.matrix(test_dtm)
  test_mat 	<- test_mat[, colnames(test_mat) %in% train_mat_cols]
  
  miss_names 	<- train_mat_cols[!(train_mat_cols %in% colnames(test_mat))]
  if(length(miss_names)!=0){
    colClasses  <- rep("numeric", length(miss_names))
    df 			<- read.table(text = '', colClasses = colClasses, col.names = miss_names)
    df[1:nrow(test_mat),] <- 0
    test_mat 	<- cbind(test_mat, df)
  }
  as.matrix(test_mat)
}

# Train and test matrices
train_mat <- chisqTwo(train_dtm, train$sentiment)
test_mat  <- testmat(colnames(train_mat), as.matrix(test_dtm))

# Take out the heavy dtms in the memory
rm(train_dtm)
rm(test_dtm)

# Run garbage collector to free up memory
gc()




library(e1071)
library(rpart)
library(mlbench)


nb <- naiveBayes(train_mat,train$sentiment)
summary(nb)
nb_pred <- predict(nb, test_mat)
table(nb_pred,test$sentiment)


svm <- svm(train_mat,train$sentiment)
summary(svm)
svm_pred <- predict(svm, test_mat)
table(svm_pred,test$sentiment)

dim(test)
dim(test_mat)

train_mat <- as.data.frame(as.matrix(train_mat))
colnames(train_mat) <- make.names(colnames(train_mat))
train_mat$sentiment <- train$sentiment


test_mat <- as.data.frame(as.matrix(test_mat))
colnames(test_mat) <- make.names(colnames(test_mat))
test_mat$sentiment <- test$sentiment



# load the library
library(mlbench)
library(caret)
# load the dataset
# prepare training scheme
control <- trainControl(method="repeatedcv", number=5, repeats=1)
# train the GBM model
set.seed(7)
modelGbm <- train(sentiment~., data=train_mat, method="gbm", trControl=control, verbose=FALSE)
# train the SVM model
set.seed(7)
modelSvm <- train(sentiment~., data=train_mat, method="svmRadial", trControl=control)
# train the LogReg model
set.seed(7)
modelLogReg <- train(sentiment~., data=train_mat, method="logreg", trControl=control)
# train the Adaboost model
set.seed(7)
modelAdaboost <- train(sentiment~., data=train_mat, method="adaboost", trControl=control)
# train the mlp model
set.seed(7)
modelmlp <- train(sentiment~., data=train_mat, method="mlp", trControl=control)
# train the NaiveBayes model
set.seed(7)
modelNB <- train(sentiment~., data=train_mat, method="naive_bayes", trControl=control)
# train the RandomForest model
set.seed(7)
modelRF <- train(sentiment~., data=train_mat, method="rf", trControl=control)


# collect resamples
results <- resamples(list(NB = modelNB, SVM=modelSvm, RF = modelRF))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)







nb.pred <- predict(modelNB,test_mat)

#Look at the confusion matrix  
confusionMatrix(nb.pred,test_mat$sentiment)   

#Draw the ROC curve 
nb.probs <- predict(modelNB,test_mat,type="prob")
head(nb.probs)



svm.pred <- predict(modelSvm,test_mat)
#Look at the confusion matrix  
confusionMatrix(svm.pred,test_mat$sentiment)   


rf.pred <- predict(modelRF,test_mat)
#Look at the confusion matrix  
confusionMatrix(rf.pred,test_mat$sentiment)   


parallelplot(results)

results$values
summary(results)

bwplot(results,metric="ROC",main="Naive Bayes vs Random Forest vs Support Vector Machine")	# boxplot
dotplot(rValues,metric="ROC",main="GBM vs xgboost")	# dotplot




