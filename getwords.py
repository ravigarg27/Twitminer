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
political_text = []
sports_text = []

def getdata():
	#Training File
	f = open('training.txt','r')
	lines=f.readlines()
	political_text_string =''
	sports_text_string = ''
	i=1
	count1=1
	for line in lines:
		flag=0
		sent = line.split()
		tweet = sent[2:]
		tweet=" ".join(tweet)
		if len(tweet.split())<=4:
			print i,tweet
			count1+=1
			flag=1
		if flag==0:
			training_tweetid.append(sent[0])
			if sent[1] == 'Sports':
				training_tweetlabel.append(0)
				sports_text_string+=tweet
			else:
				training_tweetlabel.append(1)
				political_text_string+=tweet
			training_tweettext.append(tweet)
		i+=1
	
	print 'Total Tweets',count1
	political_text=political_text_string.lower().translate(None,string.punctuation).split()
	sports_text=sports_text_string.lower().translate(None,string.punctuation).split()
	political_text_tags=political_text_string.lower().split()
	sports_text_tags=sports_text_string.lower().split()
	full_text_string = political_text+sports_text
	
	#Validation File
	validation = open('validation.txt','r')
	testlines = validation.readlines()
	test_data=''
	for testline in testlines:
		test_sent = testline.split()
		validation_tweetid.append(test_sent[0])
		tweet1 = test_sent[1:]
		tweet1 =" ".join(tweet1)
		test_data=+tweet1
		validation_tweettext.append(tweet1)
	validation.close()
	
	
		
	fdistfull = FreqDist(full_text_string)
	fdist1 = FreqDist(political_text)
	fdist2 = FreqDist(sports_text)
	fdist3 = FreqDist(political_text_tags)
	fdist4 = FreqDist(sports_text_tags)
	#print fdist1
	
	vocab_words = fdistfull.keys()
	
	vocab0 = []
	vocab1_file = open('vocab_file.txt','r')
	words = vocab1_file.readlines()
	for w in words :
		vocab0.append(w.rstrip('\n'))
	vocab1_file.close()
	
	
	vocab1 = []
	vocab2_file = open('vocab2.txt','w')
	for w in sorted(vocab_words[:]) :
		if w not in ignored_words and fdistfull[w]>=10 and w not in vocab0 and len(w)>=3:
			vocab2_file.write(w+'\n')
			vocab1.append(w)
	vocab2_file.close()
	
	
	print 'Vocab1',len(vocab1)
	print 'Vocab 0',len(vocab0)
	intersecting_words = [val for val in vocab1 if val not in vocab0]
	for w in intersecting_words:
		if fdistfull[w]<=10:
			intersecting_words.remove(w)
	print 'IntersectingWords',len(intersecting_words)
	
	vocab_try_file = open('vocab_try_file.txt','w')
	for w in sorted(vocab0):
		if fdist1[w]-fdist2[w] >=1 or fdist2[w]-fdist1[w] >=1:
			vocab_try_file.write(w+'\n')
	vocab_try_file.close()

	
####################################COLLOCATIONS########################################################
	
	collocations_file = open('collocations_words.txt','w')
	political_collocations_file = open('political_collocations_words.txt','w')
	sports_collocations_file = open('sports_collocations_words.txt','w')
	
	finder = BigramCollocationFinder.from_words(political_text)
	finder.apply_freq_filter(25)
	finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	#print len(finder.nbest(bigram_measures.raw_freq,30))
	for word in finder.nbest(bigram_measures.raw_freq,30):
		collocations_file.write(" ".join(word)+'\n')
		political_collocations_file.write(" ".join(word)+'\n')
		
		
	finder2 = BigramCollocationFinder.from_words(sports_text)
	finder2.apply_freq_filter(15)
	finder2.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	#print len(finder2.nbest(bigram_measures.raw_freq,30))
	for word in finder2.nbest(bigram_measures.raw_freq,30):
		collocations_file.write(" ".join(word)+'\n')
		sports_collocations_file.write(" ".join(word)+'\n')
	
	finder3 = TrigramCollocationFinder.from_words(political_text)
	finder3.apply_freq_filter(10)
	finder3.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	#print len(finder3.nbest(trigram_measures.raw_freq,30))
	for word in finder3.nbest(bigram_measures.raw_freq,30):
		collocations_file.write(" ".join(word)+'\n')
		political_collocations_file.write(" ".join(word)+'\n')
		
	finder4 = TrigramCollocationFinder.from_words(sports_text)
	finder4.apply_freq_filter(10)
	finder4.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
	#print len(finder4.nbest(trigram_measures.raw_freq,30)) 	
	for word in finder4.nbest(bigram_measures.raw_freq,30):
		collocations_file.write(" ".join(word)+'\n')
		sports_collocations_file.write(" ".join(word)+'\n')
####################################COLLOCATIONS########################################################	
	
	political_total_length = len(political_text)
	sports_total_length = len(sports_text)
	political_total_words = set(political_text)
	sports_total_words = set(sports_text)
	words_in_both = set(list(political_total_words & sports_total_words))
	political_unique_words = political_total_words - words_in_both
	sports_unique_words = sports_total_words - words_in_both
	
	political_hash_tags = set(list(w for w in political_text_tags if w.startswith('#')))
	sports_hash_tags = set(list(w for w in sports_text_tags if w.startswith('#')))
	both_hash_tags = set(list(political_hash_tags & sports_hash_tags))
	political_hash_tags = political_hash_tags - both_hash_tags
	sports_hash_tags = sports_hash_tags - both_hash_tags
	
	political_at_tags = set(list(w for w in political_text_tags if w.startswith('@')))
	sports_at_tags = set(list(w for w in sports_text_tags if w.startswith('@')))
	both_at_tags = set(list(political_at_tags & sports_at_tags))
	political_at_tags = political_at_tags - both_at_tags
	sports_at_tags = sports_at_tags - both_at_tags
	
	
	print 'Political Total Length',political_total_length
	print 'Sports Total Length',sports_total_length
	print 'Political Total Words',len(political_total_words)
	print 'Political Total Tags',len(set(list(w for w in political_text_tags if w.startswith('#'))))
	print 'Political at Tags',len(political_at_tags)
	print 'Sports Total Words',len(sports_total_words)
	print 'Sports Total Tags',len(set(list(w for w in sports_text_tags if w.startswith('#'))))
	print 'Sports at Tags',len(sports_at_tags)
	print 'Words in both',len(words_in_both)
	print 'Political Unique Words',len(political_unique_words)
	print 'Sports Unique Words',len(sports_unique_words)
	print 'Both Hash Tags',len(set(list(political_hash_tags & sports_hash_tags)))
	print 'Both at Tags',len(both_at_tags)
	print both_at_tags
	#print political_at_tags
	#print sports_at_tags
	
	political_at_tags_file = open('political_at_tags_file.txt','w')
	for w in political_at_tags:
		political_at_tags_file.write(w.translate(None,string.punctuation)+'\n')
	political_at_tags_file.close()
	
	sports_at_tags_file = open('sports_at_tags_file.txt','w')
	for w in sports_at_tags:
		sports_at_tags_file.write(w.translate(None,string.punctuation)+'\n')
	sports_at_tags_file.close()


	fdist1 = FreqDist(political_text)
	fdist2 = FreqDist(sports_text)
	fdist3 = FreqDist(political_text_tags)
	fdist4 = FreqDist(sports_text_tags)
	#print fdist1
	 
	f3 = open('political_word_file.txt','w')
	for w in political_unique_words:
		w = w.lower().translate(None,string.punctuation)
		if w not in stopwords.words('english'):
			freq = fdist1[w]
			#print freq
			if freq >= 50 and len(w) >= 5:
				#print "fsdfd"
				f3.write(w.lower().translate(None,string.punctuation)+'\n')
		
		
	f4 = open('sports_word_file.txt','w')
	for w in sports_unique_words:
		w = w.lower().translate(None,string.punctuation)
		if w not in stopwords.words('english'):
			freq = fdist2[w]
			#print type(freq)
			if freq >= 50 and len(w)>=6:
				#print freq
				f4.write(w.lower().translate(None,string.punctuation)+'\n')
			
	f5 = open('both_word_file.txt','w')
	for w in words_in_both:
		w = w.lower().translate(None,string.punctuation)
		if w not in stopwords.words('english'):
			freq2 = fdist2[w]
			freq1 = fdist1[w]
			diff=max(freq1-freq2,freq2-freq1)
			if diff >= 50 and len(w) >= 6:
				if freq1>freq2:
					f3.write(w+'\n')
				else:
					f4.write(w+'\n')
				f5.write(w+' '+'Sports'+str(freq2)+' '+'Political'+str(freq1)+'\n')
				
	
	f3.close()
	f4.close()
	f5.close()	
	
	###############HASHTAGS######################
	f6 = open('political_hash_tags.txt','w')	
	for w in set(political_hash_tags):
		w=w.lower().translate(None,string.punctuation)
		freq = fdist1[w]
		if freq >=20 and len(w)>=5:
			f6.write('#'+w+'\n')
	f6.close()
	
	f7 = open('sports_hash_tags.txt','w')
	for w in set(sports_hash_tags):
		w=w.lower().translate(None,string.punctuation)
		freq = fdist2[w]
		if freq >=10 and len(w)>=5:
			f7.write('#'+w+'\n')
	f7.close()
	
	f.close()
	
	
	
if __name__ == "__main__":
	getdata()
	
