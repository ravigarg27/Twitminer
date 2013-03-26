from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC,NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
import string
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
ignored_words = nltk.corpus.stopwords.words('english')

training_tweetid = []
training_tweetlabel = []
training_tweettext = []
validation_tweetid = []
validation_tweettext = []
training_text = []
test_text = []
political_words=['government','refugee','congress','samajwadi','reform','speech','diplomacy','staff','defence','educate','finance','money','bjp','political','corruption']
sports_words=['team','sport','medal','tabletennis','paralympics','athlete','game','boxing','baseball','tennis','singles','mixed doubles','match','final','singles','championship','pingpong','semi final','usopen','wimbledon','training','exercise','bat','ball','racquet','goal','championship','chess','club','league','youtube']
political_at_list=[]
sports_at_list=[]

def getdata():
	#Training File
	f = open('training1.txt','r')
	lines=f.readlines()
	i=1
	count1=1
	for line in lines:
		flag=0
		sent = line.split()
		tweet = sent[2:]
		tweet=" ".join(tweet)
		if len(tweet.split())<=2 and '#' not in tweet:
			#print i,tweet
			count1+=1
			flag=1
		if flag==0:
			training_tweetid.append(sent[0])
			if sent[1] == 'Sports':
				training_tweetlabel.append(0)
				#	sports_text_string+=tweet
			else:
				training_tweetlabel.append(1)
					#political_text_string+=tweet
			training_tweettext.append(tweet)
		i+=1
	
	print 'Tweets',count1
	f.close()
	
	#Validation File
	validation = open('test_final.txt','r')
	testlines = validation.readlines()
	for testline in testlines:
		test_sent = testline.split()
		validation_tweetid.append(test_sent[0])
		tweet1 = test_sent[1:]
		tweet1 =" ".join(tweet1)
		validation_tweettext.append(tweet1)
	validation.close()
	
	
	#get @user
	political_at_tags_file = open('political_at_tags_file.txt','r')
	tags = political_at_tags_file.readlines()
	for tag in tags:
		#print tag
		political_words.append(tag)
	political_at_tags_file.close()
	
	sports_at_tags_file = open('sports_at_tags_file.txt','r')
	tags = sports_at_tags_file.readlines()
	for tag in set(tags):
		#print tag
		sports_words.append(tag)
	sports_at_tags_file.close()
	
	#get hash tags
	
	political_hash_tags_file = open('political_hash_tags.txt','r')
	tags = political_hash_tags_file.readlines()
	for tag in tags:
		political_words.append(tag.rstrip('\n'))
	political_hash_tags_file.close()
	
	sports_hash_tags_file = open('sports_hash_tags.txt','r')
	tags = sports_hash_tags_file.readlines()
	for tag in set(tags):
		sports_words.append(tag.rstrip('\n'))
	sports_hash_tags_file.close()
	
	sports_collocations_file = open('sports_collocations_words.txt','r')
	words = sports_collocations_file.readlines()
	for word in set(words):
		sports_words.append(word.rstrip('\n'))
	sports_collocations_file.close()
	
	political_collocations_file = open('political_collocations_words.txt','r')
	words = political_collocations_file.readlines()
	for word in set(words):
		political_words.append(word.rstrip('\n'))
	political_collocations_file.close()
	
#	political_word_file_new = open('political_word_file.txt','r')
#	words = political_word_file_new.readlines()
#	for word in set(words):
#		political_words.append(word.rstrip('\n'))
#	political_word_file_new.close()
#	
#	sports_word_file_new = open('sports_word_file.txt','r')
#	words = sports_word_file_new.readlines()
#	for word in set(words):
#		sports_words.append(word.rstrip('\n'))
#	sports_word_file_new.close()	
	
	both_words = list(set(political_words) & set(sports_words))
	print len(both_words)
	print 'Political Words Length',len(political_words)
	print 'Sports Words Length',len(sports_words)

	
def preprocess():
	vocab = []
	vocab_file = open('vocab_try_file.txt','r')
	words = vocab_file.readlines()
	for w in words:
		vocab.append(w.rstrip('\n'))
	vocab_file.close()
	
	count_vect = CountVectorizer(min_df=1,max_df=0.4)
	counts = count_vect.fit_transform(training_tweettext)
	#print type(count_vect)
	#print type(counts)
	print counts.shape
	
#	vocab_write_file = open('vocab_file.txt','w')
#	for w in sorted(count_vect.vocabulary_):
#		vocab_write_file.write(w+'\n')
#	vocab_write_file.close()

	tfidf_transformer = TfidfTransformer()
	train_tfidf = tfidf_transformer.fit_transform(counts).toarray()
	test_counts = count_vect.transform(validation_tweettext)
	test_tfidf = tfidf_transformer.transform(test_counts).toarray()

	clf = MultinomialNB(alpha=0.25).fit(train_tfidf, training_tweetlabel)	
	predicted = clf.predict(test_tfidf)
	probability = clf.predict_proba(test_tfidf)
	result = open('result.txt','w')

	
###################Data for iter1#####################################
	new_train_data = []
	new_train_label =[]
	new_test_data = []
	new_test_data_id = []
	old_label=[]
	probability_file = open('probability.txt','w')
	file_for_iter_2 = open('file_for_iter_2.txt','w')
	count_tweet=0
	for i in probability:
		if i[0] >= 0.80:
			new_train_data.append(validation_tweettext[count_tweet])
			new_train_label.append(0)
			
		elif i[0] <= 0.20:
			new_train_data.append(validation_tweettext[count_tweet])
			new_train_label.append(1)
		else:
			new_test_data.append(validation_tweettext[count_tweet])
			file_for_iter_2.write(validation_tweettext[count_tweet]+'\n')
			new_test_data_id.append(count_tweet+1)     ###########Storing the Line Number of Test Id
			if i[0]>i[1]:
				probability_file.write(str(i)+'Sports'+str(count_tweet+1)+'\n')
				old_label.append('Sports')
			else:
				probability_file.write(str(i)+'Politics'+str(count_tweet+1)+'\n')
				old_label.append('Politics')
		count_tweet+=1
	probability_file.close()
	file_for_iter_2.close()
############################Iter1Data Over###########################################

	political_data_for_words = []
	sports_data_for_words = []
	for i in range(len(new_train_data)):
		if new_train_label[i] ==0:
			sports_data_for_words.append(new_train_data[i])
		elif new_train_label[i] ==1:
			political_data_for_words.append(new_train_data[i])
			
#############################Iter1MODEL#########################################
	count_vect_new = CountVectorizer(min_df=1,max_df=0.4)
	new_train_data = new_train_data + training_tweettext
	new_train_label = new_train_label+training_tweetlabel
	counts_new = count_vect_new.fit_transform(new_train_data)
	print counts_new.shape
	tfidf_transformer_new = TfidfTransformer()
	train_tfidf_new = tfidf_transformer_new.fit_transform(counts_new).toarray()
	test_counts_new = count_vect_new.transform(new_test_data)
	test_tfidf_new = tfidf_transformer_new.transform(test_counts_new).toarray()
		
	clf_new = MultinomialNB(alpha=0.25).fit(train_tfidf_new, new_train_label)	
	predicted_new_test = clf_new.predict(test_tfidf_new)
	probability_new_test = clf_new.predict_proba(test_tfidf_new)
	
	new_label = []
	new_test_data_label = []
	
	result_new = open('result_new.txt','w')
	for i in range(len(predicted_new_test)):
		if predicted_new_test[i] == 0:
			new_label.append('Sports')
			new_test_data_label.append(0)
			result_new.write('Sports'+'\n')
		else:
			new_label.append('Politics')
			new_test_data_label.append(1)
			result_new.write('Politics'+'\n')
	result_new.close()
		
	corrected_ones=0
	for i in range(len(new_label)):
		if new_label[i] != old_label[i]:
			corrected_ones+=1
			print old_label[i],i+1,new_test_data_id[i],predicted[new_test_data_id[i]-1]
			if predicted[new_test_data_id[i]-1] == 0:
				predicted[new_test_data_id[i]-1] =1
			elif predicted[new_test_data_id[i]-1] == 1:
				predicted[new_test_data_id[i]-1] =0
	print 'Corrected Ones',corrected_ones

#############################Iter1OVER########################################################################
#############################Iter2DATA########################################################################
	count_iter_two=1			#########Only for Checking Purpose
	for i in probability_new_test:
		if i[0]>=0.40 and i[0] <= 0.60:
			#print 'Probability 2 iter',i
			count_iter_two+=1
	print 'Count for iter two',count_iter_two
#		
#	print 'Old Test data',len(new_test_data)                       ####All 1004
#	print 'Old Test data label',len(new_test_data_label)
#	print 'Old Probability Length',len(probability_new_test)
	
	new_test_data_iter2 = []
	probability_file_iter2=open('probability_file_iter2.txt','w')
	count_tweet=0
	old_label_iter2=[]
	new_test_data_iter2_id=[]
	for i in probability_new_test:
		if i[0] >= 0.55:
			new_train_data.append(new_test_data[count_tweet])
			new_train_label.append(new_test_data_label[count_tweet])
			sports_data_for_words.append(new_test_data[count_tweet])
			
		elif i[0] <= 0.45:
			new_train_data.append(new_test_data[count_tweet])
			new_train_label.append(new_test_data_label[count_tweet])
			political_data_for_words.append(new_test_data[count_tweet])
		else:
			new_test_data_iter2.append(new_test_data[count_tweet])
			new_test_data_iter2_id.append(new_test_data_id[count_tweet])
			if i[0]>i[1]:
				probability_file_iter2.write(str(i)+'Sports'+str(new_test_data_id[count_tweet])+'\n')  #P,L,LineNo
				old_label_iter2.append('Sports')
				
			else:
				probability_file_iter2.write(str(i)+'Politics'+str(new_test_data_id[count_tweet])+'\n')
				old_label_iter2.append('Politics')
		count_tweet+=1
		
	#print 'Training Data For Iter2',len(new_train_data)
	#print 'Training Label For Iter2',len(new_train_label)
	#print 'Testing Data For Iter1',len(new_test_data_iter2)
	
########################Iter2DATAOVER############################################################################
########################Iter2Model###############################################################################
	count_vect_new = CountVectorizer(min_df=1,max_df=0.4)
	counts_new_iter2 = count_vect_new.fit_transform(new_train_data)
	print counts_new_iter2.shape
	tfidf_transformer_new = TfidfTransformer()
	train_tfidf_new_iter2 = tfidf_transformer_new.fit_transform(counts_new_iter2).toarray()
	
	test_counts_new_iter2 = count_vect_new.transform(new_test_data_iter2)
	test_tfidf_new_iter2 = tfidf_transformer_new.transform(test_counts_new_iter2).toarray()
	
	
	clf_new_iter2 = MultinomialNB(alpha=0.25).fit(train_tfidf_new_iter2, new_train_label)	####Prediction On 2ndIteration Test Data
	predicted_new_test_iter2 = clf_new_iter2.predict(test_tfidf_new_iter2)
	probability_new_test_iter2 = clf_new.predict_proba(test_tfidf_new)
	
	
	new_label_iter2=[]
	for i in range(len(predicted_new_test_iter2)):							#####Get The Labels From 2nd Iter
		if predicted_new_test_iter2[i]==0:								
			new_label_iter2.append('Sports')
		else:
			new_label_iter2.append('Politics')
			
	print 'Length of old label for iter2',len(old_label_iter2)
	print 'Length of new label for iter2',len(new_label_iter2)
	
	#########Intersection Or The Corrected Ones####################
	#print 'Lenght new',len(new_test_data_iter2_id)
	corrected_ones_iter2=0
	for i in range(len(new_label_iter2)):
		if new_label_iter2[i]!=old_label_iter2[i]:		
			corrected_ones_iter2+=1
			print predicted[new_test_data_iter2_id[i]-1],new_label_iter2[i],old_label_iter2[i],new_test_data_iter2_id[i]
			if predicted[new_test_data_iter2_id[i]-1] == 0:
				predicted[new_test_data_iter2_id[i]-1]=1
			elif predicted[new_test_data_iter2_id[i]-1] == 1:
				predicted[new_test_data_iter2_id[i]-1]=0
	
	print 'Corrected Ones for Iter2',corrected_ones_iter2

#########################Iter2ModelOVER#################################################################################	
	clf2 = LinearSVC(C=10).fit(train_tfidf_new_iter2, new_train_label)	
	predicted1 = clf.predict(test_tfidf)
	result1 = open('resultpython11.txt','w')
	count = []

##############################GetWordsFromNewTestDataAlso###############################################################

	print 'Political data for words',len(political_data_for_words)
	print 'Sports data for words',len(sports_data_for_words)
	
	political_data_for_words_text=''                   #####Stores The text format
	for tweet in political_data_for_words:
		political_data_for_words_text+=tweet
		political_data_for_words_text+=' '
	political_data_for_words = political_data_for_words_text.lower().translate(None,string.punctuation).split()
	###Store in List format each word
		
	sports_data_for_words_text=''
	for tweet in sports_data_for_words:
		sports_data_for_words_text+=tweet
		sports_data_for_words_text+=' '
	sports_data_for_words = sports_data_for_words_text.lower().translate(None,string.punctuation).split()
	
	###########################GetTheWords########################################################
	fdist1 = FreqDist(political_data_for_words)
	fdist2 = FreqDist(sports_data_for_words)
	
	political_total_length = len(political_data_for_words)
	sports_total_length = len(sports_data_for_words)
	political_total_words = set(political_data_for_words)
	sports_total_words = set(sports_data_for_words)
	words_in_both = set(list(political_total_words & sports_total_words))
	political_unique_words = political_total_words - words_in_both
	sports_unique_words = sports_total_words - words_in_both
	
	print 'Political Total Length',political_total_length
	print 'Sports Total Length',sports_total_length
	print 'Political Total Words',len(political_total_words)
	print 'Sports Total Words',len(sports_total_words)
	print 'Words in both',len(words_in_both)
	print 'Political Unique Words',len(political_unique_words)
	print 'Sports Unique Words',len(sports_unique_words)
	
	for w in political_unique_words:
		if w not in stopwords.words('english'):
			freq = fdist1[w]
			if freq >= 25 and len(w) >=4:
				print 'Political Word To Be Added',w
				political_words.append(w)
				
	for w in sports_unique_words:
		if w not in stopwords.words('english'):
			freq = fdist2[w]
			if freq >= 25 and len(w)>5:
				print 'Sports Word To Be Added',w
				sports_words.append(w)
			
	for w in words_in_both:
		if w not in stopwords.words('english'):
			freq2 = fdist2[w]
			freq1 = fdist1[w]
			diff=max(freq1-freq2,freq2-freq1)
			if diff >= 30 and len(w) >= 7:
				if freq1>freq2:
					print 'Political Word To Be Added',w
					political_words.append(w)
				else:
					print 'Sports Word To Be Added',w
					sports_words.append(w)
	
	print 'Length of political words new before collocations',len(political_words)
	print 'Length of sports words new before collocations',len(sports_words)							
	
	#############################GetTheSingleWordsOver#################################################
	
########################Collocations##########################################
	finder = BigramCollocationFinder.from_words(political_data_for_words)
	finder.apply_freq_filter(10)
	finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	print 'Length of Bigram Political',len(finder.nbest(bigram_measures.raw_freq,30))
	for word in finder.nbest(bigram_measures.raw_freq,30):
		w = ' '.join(word)
		political_words.append(w)
		print w
			
	finder2 = BigramCollocationFinder.from_words(sports_data_for_words)
	finder2.apply_freq_filter(10)
	finder2.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	print 'Length of Bigram Sports',len(finder2.nbest(bigram_measures.raw_freq,30))
	for word in finder2.nbest(bigram_measures.raw_freq,30):
		w = ' '.join(word)
		sports_words.append(w)
		print w
	
	finder3 = TrigramCollocationFinder.from_words(political_data_for_words)
	finder3.apply_freq_filter(8)
	finder3.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	print 'Length of Trigram Political',len(finder3.nbest(trigram_measures.raw_freq,30))
	for word in finder3.nbest(trigram_measures.raw_freq,30):
		w = ' '.join(word)
		political_words.append(w)
		print w
		
	finder4 = TrigramCollocationFinder.from_words(sports_data_for_words)
	finder4.apply_freq_filter(8)
	finder4.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	print 'Length of Trigram Sports',len(finder4.nbest(trigram_measures.raw_freq,30))	
	for word in finder4.nbest(trigram_measures.raw_freq,30):
		w = ' '.join(word)
		sports_words.append(w)
		print w

	print 'Length of political words new After Collocations',len(political_words)
	print 'Length of sports words new After Collocations',len(sports_words)

	#######################CollocationsOver#######################################################
	#########################AtTheRateTagsStart###################################################
	fdist3 = FreqDist(political_data_for_words_text.lower().split())
	fdist4 = FreqDist(sports_data_for_words_text.lower().split())
	
	political_at_tags = set(list(w for w in political_data_for_words_text.lower().split() if w.startswith('@')))
	sports_at_tags = set(list(w for w in sports_data_for_words_text.lower().split() if w.startswith('@')))
	both_at_tags = set(list(political_at_tags & sports_at_tags))
	political_at_tags = political_at_tags - both_at_tags
	sports_at_tags = sports_at_tags - both_at_tags
	
	political_hash_tags = set(list(w for w in political_data_for_words_text.lower().split() if w.startswith('#')))
	sports_hash_tags = set(list(w for w in sports_data_for_words_text.lower().split() if w.startswith('#')))
	both_hash_tags = set(list(political_hash_tags & sports_hash_tags))
	political_hash_tags = political_hash_tags - both_hash_tags
	sports_hash_tags = sports_hash_tags - both_hash_tags
	
	print 'Len of political hash tags',len(political_hash_tags)
	print 'Len of sports hash tags',len(sports_hash_tags)
	print 'Len of both hash tags',len(both_hash_tags)
	
	print 'Len of political at tags',len(political_at_tags)
	print 'Len of sports at tags',len(sports_at_tags)
	print 'Len of both at tags',len(both_at_tags)
	
	for w in political_hash_tags:
		if fdist3[w] > 12 and w not in ignored_words and len(w)>=4:
			print w
			political_words.append(w.translate(None,string.punctuation))
	for w in political_at_tags:
		if fdist3[w] > 12 and w not in ignored_words and len(w)>=4:
			print w
			political_words.append(w.translate(None,string.punctuation))
	for w in sports_hash_tags:
		if fdist4[w] > 12 and w not in ignored_words and len(w)>=4:
			print w
			sports_words.append(w.translate(None,string.punctuation))
	for w in sports_at_tags:
		if fdist4[w] > 12 and w not in ignored_words and len(w)>=4:
			print w
			sports_words.append(w.translate(None,string.punctuation))
	
	print 'Length finally',len(political_words)
	print 'Length finally',len(sports_words)
	
	##########################AtTheRateTagsOver#####################################################
#################################GettingWordsOver#########################################################################
##########################WritingTheResults##############################################################################	
	for i in range(len(predicted)):
		flag=0
		flag2=0
		resultline = ''
		count_political=0
		count_sports=0
		for word in political_words:
			if word in validation_tweettext[i].lower().translate(None,string.punctuation):
				count_political+=1
				
		for word in sports_words:
			if word in validation_tweettext[i].lower().translate(None,string.punctuation):
				count_sports+=1
				
		if count_political != 0 and predicted[i] ==0:	#i.e Sports
			if count_political > count_sports:
				count.append(i+1)
				resultline = validation_tweetid[i]+' '+'Politics'+'\n'
			else:
				resultline = validation_tweetid[i]+' '+'Sports'+'\n'
		elif count_sports != 0 and predicted[i]==1:	#i.e Politics
			if count_sports > count_political:
				count.append(i+1)
				resultline = validation_tweetid[i]+' '+'Sports'+'\n'
			else:
				resultline = validation_tweetid[i]+' '+'Politics'+'\n'
		else:
			if predicted[i] == 0:
				resultline = validation_tweetid[i]+' '+'Sports'+'\n'
				#print 3
			else:
				resultline = validation_tweetid[i]+' '+'Politics'+'\n'
				#print 4
#		for word in political_words:
#			if word in validation_tweettext[i].lower().translate(None,string.punctuation) and predicted[i] == 0:
#				print 'Sports',i+1,word
#				count.append(i+1)
#				resultline = validation_tweetid[i]+' '+'Politics'+'\n'
#				flag=1
#				flag2=1
#				#print 1
#				break
#			
#				
#		#print '################'
#		for word in sports_words:
#			if word in validation_tweettext[i].lower().translate(None,string.punctuation) and predicted[i] == 1 and flag==0:
#				print i+1,word
#				count.append(i+1)
#				
#				resultline = validation_tweetid[i]+' '+'Sports'+'\n'
#				flag2=1
#				break
#				#print 2
#		
#				
#		if flag2==0:
#			if predicted[i] == 0:
#				resultline = validation_tweetid[i]+' '+'Sports'+'\n'
#				#print 3
#			else:
#				resultline = validation_tweetid[i]+' '+'Politics'+'\n'
#				#print 4
		result.write(resultline)
	result.close()
	
	print 'Number',len(count)
	
	for i in range(len(predicted1)):
		resultline1 = ''
		if predicted1[i] == 0:
			resultline1 = validation_tweetid[i]+' '+'Sports'+'\n'
		else:
			resultline1 = validation_tweetid[i]+' '+'Politics'+'\n'
		result1.write(resultline1)
	result1.close()
##########################WritingTheResults##############################################################################	
if __name__ == "__main__":
	getdata()
	preprocess()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
