import re
import numpy as np
from statistics import harmonic_mean as hm
import math
import pandas as pd
from math import log10

# function to compute average from a list
def get_average(l):
  return sum(l)/len(l)

# READING FROM FILE
file = open('brown.txt', "r")
if file:
	text = file.read()
lines = text.split("\n")
lines=lines[:15]

# PRE-PROCESSING THE DATA
processed_lines = []
for i in lines:
  processed_tokens=[]
  tokens = i.split(" ")
  if len(tokens)>1:
    for j in tokens:
      processed_word_tag="######"
      word_tag = j.split("_")
      if len(word_tag)==2:
        tag = re.split("-HL|-TL|-NC",word_tag[1])[0]
        tag = tag.split("FW-")[-1]
        processed_word_tag = word_tag[0].lower()+"_"+tag
        processed_tokens.append(processed_word_tag)
    start = ["^_^"]
    end = ["._."]
    sent = start + processed_tokens
    if processed_tokens[-1]!=end[0]:
      sent+=end
    processed_lines.append(sent)
  
# Splitting the data  
length = len(processed_lines)
mod = length%3
part = int((length-mod)/3)
part1 = processed_lines[:part+mod]
part2 = processed_lines[part+mod:(2*part)+mod]
part3 = processed_lines[(2*part)+mod:]
parts = [part1,part2,part3]

#===========================================================
#					CROSS - VALIDATION STARTS HERE
#							MARKOV LENGTH 1
#===========================================================
matrices_across_folds = {}
overall_matrices_across_folds = {}
frequnecy_distribution_across_folds = {}
percentage_distribution_across_folds = {}
ambiguous_matrix_across_folds = {}

for phase in range(3): # 0,1,2
	train = parts[(phase)%3] + parts[(phase+1)%3]
	test = parts[(phase+2)%3]
	# train = part1 + part2
	# test = part3

	#===========================================================
	#					TRAINING
	#===========================================================

	# PRE-PROCESSING OF TOKENS
	tag_set = set([])  # all the tags from train data
	word_set = set([]) # all words from train data
	for i in train:
		tokens = i
		for j in tokens:
			word_tag = j.split("_")
			tag_set.add(word_tag[1])
			word_set.add(word_tag[0]) 

	tags = list(tag_set)
	template_tags = []
	tags_index = {}
	l = len(tags)
	for i in range(l):
		template_tags.append(0)
		tags_index[tags[i]] = i

	words = list(word_set)
	template_words = []
	words_index = {}
	l = len(words)
	for i in range(l):
		template_words.append(0)
		words_index[words[i]] = i

	# FINDING EMISSION AND TRANSITION COUNTS
	emission_counts = {} # tag : [countS corresponding to all words]
	# total_counts_for_all_tags = template_tags[:] #[count of occurence of all tags]
	total_counts_for_all_tags = {} 

	for i in tags:
	  emission_counts[i] = template_words[:]

	total_trans = 0 
	transition_counts = {} # {state1:  {state2 : count }}

	ambiguity={}
	occurences={}

	for i in tags:
	  temp = {}
	  for j in tags:
	    temp[j]=0
	  transition_counts[i]=temp

	for i in train:
	  tokens = i
	  for j in range(len(tokens)-1):
	    token = tokens[j]
	    word_tag = token.split("_")
	    s1 = word_tag[1]
	    token = tokens[j+1]
	    word_tag = token.split("_")
	    s2 = word_tag[1]
	    total_trans += 1
	    transition_counts[s1][s2] += 1

	for i in train:
	  tokens = i
	  for j in tokens:
	    word_tag = j.split("_")
	    word = word_tag[0]
	    tag = word_tag[1]
	    index = words_index[word]
	    emission_counts[tag][index] += 1
	    index = tags_index[tag]
	    total_counts_for_all_tags[tag] = total_counts_for_all_tags.get(tag,0)+1
	    if word in occurences:
	      occurences[word]+=1
	      ambiguity[word].add(tag)
	    else:
	      occurences[word]=1
	      ambiguity[word]=set([tag])

	#===========================================================
	#					TESTING DATA IS SEEN HERE
	#===========================================================
	new_words = set([]) # have zero occurence counts
	new_tags = set([]) # have zero occurence counts

	# SEPARATING WORDS AND TAGS OF THE TEST DATA
	test_data=[]
	actual_tags = []
	for i in test:
	  tokens = i
	  words=[]
	  tags=[]
	  for j in tokens:
	    word_tag = j.split("_")
	    word = word_tag[0]
	    tag = word_tag[1]
	    words.append(word)
	    tags.append(tag)
	    if word not in word_set:
	      new_words.add(word)
	    if tag not in tag_set:
	      new_tags.add(tag)
	  test_data.append(words)
	  actual_tags.append(tags)

	# FINDING TOTAL SIZES OF VOCABULARIES
	all_words = set(list(word_set)+list(new_words))
	vocab_of_words = len(all_words)
	all_tags = set(list(tag_set)+list(new_tags))
	vocab_of_tags = len(all_tags)
	# transition_size = (len(all_tags))**2

	#===========================================================
	#		CALCULATING PROBABILITIES WITH SMOOTHING
	#===========================================================
	total_tag_counts = sum(list(total_counts_for_all_tags.values()))

	# PROBABILITY OF OCCURENCE OF A TAG
	tag_probabs = {} 
	for k in emission_counts.keys(): # k = tag
	  v = sum(emission_counts[k])
	  tag_probabs[k] = (v+1)/(total_tag_counts + vocab_of_tags)
	for i in new_tags:
	  tag_probabs[i] = 1/(total_tag_counts + vocab_of_tags)

	# PROBABILITY OF A WORD GIVEN A TAG IN LOG SPACE
	word_given_tag_probabs={}  #tag: {word : count}
	for i in tag_set:
	  word_probabs={}
	  for j in word_set:
	    index = words_index[j]
	    word_probabs[j] = log10((emission_counts[i][index] +1)/(tag_probabs[i]*total_tag_counts + vocab_of_words))
	  for j in new_words:
	    word_probabs[j] = log10(1/(tag_probabs[i]*total_tag_counts + vocab_of_words))
	  word_given_tag_probabs[i] = word_probabs

	for i in new_tags:
	  word_probabs={}
	  for j in all_words:
	    word_probabs[j] = log10(1/(tag_probabs[i]*total_tag_counts + vocab_of_words))
	  word_given_tag_probabs[i] = word_probabs


	# PROBABILITY OF A STATE(SAY 2) GIVEN PREVIOUS STATE(SAY 1) IN LOG SPACE
	s2_given_s1_probabs={} #tag2: {tag1 : count}
	for t2 in tag_set:
	  t1_probabs={}
	  for t1 in tag_set:
	    t1_probabs[t1] = log10((transition_counts[t1][t2] + 1)/(tag_probabs[t1]*total_tag_counts + total_trans))
	  for t1 in new_tags:
	    t1_probabs[t1] = log10(1/(tag_probabs[t1]*total_tag_counts + total_trans))
	  s2_given_s1_probabs[t2] = t1_probabs

	for t2 in new_tags:
	  t1_probabs={}
	  for t1 in all_tags:
	    t1_probabs[t1] = log10(1/(tag_probabs[t1]*total_tag_counts + total_trans))
	  s2_given_s1_probabs[t2] = t1_probabs

	#===========================================================
	#			CALCULATING STATS OF TAGS IN VOCAB
	#===========================================================

	for tag in new_tags:
	  total_counts_for_all_tags[tag] = 0 #frequencies

	print("Statistics of tags: frequency distribution:")
	print(total_counts_for_all_tags)
	frequnecy_distribution_across_folds[phase] = total_counts_for_all_tags

	percentages={}
	for i in all_tags:
	  percentages[i] = (total_counts_for_all_tags[i] * 100)/total_tag_counts
	print("Statistics of tags: percentage distribution:")
	print(percentages)
	percentage_distribution_across_folds[phase] = percentages


	ambiguous={}
	for k in ambiguity.keys():
	  if len(ambiguity[k])>1:
	    ambiguous[k]=len(ambiguity[k])

	total_occurences = sum(list(occurences.values()))
	ambiguous_occurences={}
	for k in ambiguous.keys():
	  ambiguous_occurences[(k,ambiguous[k])] = (100*occurences[k])/total_occurences
	# print("Words | Tags | Percentage of occurences")
	# print(ambiguous_occurences)
	a=(len(ambiguous)*100)/len(all_words)
	b=sum(list(ambiguous_occurences.values()))
	print("Percentage of words that are ambiguous")
	print(a)
	print("Total occurences percentage of ambiguous words")
	print(b)
	ambiguous_matrix_across_folds[phase] = [a,b]

	#===========================================================
	#					TESTING BEGINS
	#===========================================================
	results=[]
	for i in test_data:
	  result=[]
	  tokens = i
	  entry_tag = tokens[0] # THIS IS ALWAYS ^
	  probabilities = {} 
	  for t in all_tags:
	    probabilities[t] = s2_given_s1_probabs[t][entry_tag] # PROBABILITY OF TAG(i) GIVEN PREVIOUS TAG WAS ^

	  # TO FIND P(s|o)
	  for j in range(len(tokens)):
	    old_word = tokens[j]
	    # GIVES P(o|s)
	    probab_word_given_tags={}
	    for t in all_tags:
	      probab_word_given_tags[t] = word_given_tag_probabs[t][old_word]

	    # P(s2|s1) ARE ALREADY CALCULATED AND STORED IN s2_given_s1_probabs
	    P = {} # {s1 : p(o1|s1)*p(older(till s1))}
	    for k in probabilities.keys(): 
	      tags = k.split(" ")
	      last_tag = tags[-1] # gives s1
	      P[last_tag] = probabilities[k] + probab_word_given_tags[last_tag]
	      
	    # COMPUTING BEST PATH REACHING EACH TAG    
	    next_probabilities={} # {best path to tag : probability}
	    for t2 in all_tags:
	      probabs = {} # probabilites of all paths reaching t2
	      for k in probabilities.keys():
	        v = probabilities[k]
	        l = k.split(" ")
	        t1 = l[-1]
	        probabs[k] = P[t1] + s2_given_s1_probabs[t2][t1]
	    
	      key_max = max(probabs, key=probabs.get) # max probability, best path
	      value_max = probabs[key_max]
	      l = key_max.split(" ")
	      l.append(t2) # t2 is added to choosen path
	      v = value_max 
	      k = " ".join(i for i in l)
	      next_probabilities[k]=v
	    probabilities = next_probabilities
	    
	  key_max = max(probabilities, key=probabilities.get) # best tag set
	  result = key_max
	  results.append(result.split(" ")[:-1])
	print(results)
	# for i in range(5):
	#   print(test_data[i])
	#   print(actual_tags[i])
	#   print(results[i])
	# print("--- %s seconds ---" % (time.time() - start_time))

	#===========================================================
	#				CREATING CONFUSION MATRIX
	#===========================================================

	all_tags_index = {}
	all_tags_list = list(all_tags)
	all_tags_list.sort()
	for i in range(len(all_tags)):
	  all_tags_index[all_tags_list[i]] = i

	dimension = len(all_tags)
	confusion_matrix = np.zeros((dimension,dimension))

	for i in range(len(actual_tags)):
	  for j in range(len(actual_tags[i])):
	    at = actual_tags[i][j]
	    pt = results[i][j]
	    ati = all_tags_index[at]
	    pti = all_tags_index[pt]
	    confusion_matrix[ati][pti]+=1

	total = np.sum(confusion_matrix)
	predicted_totals = np.sum(confusion_matrix, axis = 0)
	actual_totals = np.sum(confusion_matrix, axis = 1)

	#===========================================================
	#				FINDING TP, FP,FN FOR ALL TAGS
	#===========================================================

	# TP = matrix(tag,tag)
	# FP = predicted - TP
	# FN = actual - TP
	# TN = total - actual - predicted + TP

	tag_scores = {} # tag : {TP: count , FP: count, TN: count, FN: count}
	for tag in all_tags:
	  index = all_tags_index[tag] # tag to an index
	  scores={}
	  tp = confusion_matrix[index][index]
	  scores["TP"] = tp
	  fp = predicted_totals[index] - tp
	  scores["FP"] = fp
	  fn = actual_totals[index] - tp
	  scores["FN"] = fn
	  # tn = total - actual_totals[index] - predicted_totals[index] + tp
	  # scores["TN"] = tn
	  tag_scores[tag] = scores

	#===========================================================
	#	CALCULATING TAG-WISE PRECISION, RECALL, F1-SCORE
	#===========================================================
	matrices={}
	for tag in all_tags:
	  c = tag_scores[tag]
	  precision = c["TP"]/(c["TP"]+c["FP"])
	  precision = 0 if math.isnan(precision) else precision
	  recall = c["TP"]/(c["TP"]+c["FN"])
	  recall = 0 if math.isnan(recall) else recall
	  temp = [precision,recall]
	  f1_score = hm(temp)
	  matrices[tag] = [precision,recall, f1_score]

	  if tag in matrices_across_folds:
	    matrices_across_folds[tag][phase] = [precision,recall, f1_score]
	  else:
	    matrices_across_folds[tag] = {phase:[precision,recall, f1_score]}

  	# except KeyError as ke:
	# 	matrices_across_folds
	print("Tag-wise precision, recall and f1-score")
	print(matrices)
	# length = len(matrices_across_folds)
	# matrices_across_folds[length] = matrices

	# for k in list(matrices.keys())[:5]:
	#   print(k,matrices[k])

	#===========================================================
	#	CALCULATING PRECISION, RECALL, F1-SCORE
	#===========================================================
	overall_matrices=[]
	all_precisions = []
	all_recalls=[]
	all_f1_scores=[]
	for k in matrices.keys():
	  a = matrices[k]
	  all_precisions.append(a[0])
	  all_recalls.append(a[1])
	  all_f1_scores.append(a[2])

	overall_matrices.append(get_average(all_precisions))
	overall_matrices.append(get_average(all_recalls))
	overall_matrices.append(get_average(all_f1_scores))

	print("Overall precision, recall and f1-score")
	print(overall_matrices)
	# length = len(overall_matrices_across_folds)
	overall_matrices_across_folds[phase] = overall_matrices

#===========================================================
#					3 FOLDS END HERE
#===========================================================

tagwise_average_scores = {}
for k in matrices_across_folds.keys(): # k = tag
	phase_wise = list(matrices_across_folds[k].values()) #[[p1,r1,f1],[p2,r2,f2],[p3,r3,f3]]
	avg_pre = 0
	avg_rec = 0
	avg_f1 = 0
	for i in phase_wise:
		avg_pre+=i[0]
		avg_rec+=i[1]
		avg_f1+=i[2]
	avg_pre/=3
	avg_rec/=3
	avg_f1/=3
	tagwise_average_scores[k] = [avg_pre,avg_rec,avg_f1]

average_scores = []
phase_wise = list(overall_matrices_across_folds.values()) #[[p1,r1,f1],[p2,r2,f2],[p3,r3,f3]]
avg_pre = 0
avg_rec = 0
avg_f1 = 0
for i in phase_wise:
	avg_pre+=i[0]
	avg_rec+=i[1]
	avg_f1+=i[2]
avg_pre/=3
avg_rec/=3
avg_f1/=3
average_scores = [avg_pre,avg_rec,avg_f1]

tagwise_average_frequency={}
for k in frequnecy_distribution_across_folds.keys():
	v = frequnecy_distribution_across_folds[k]
	for k2 in v.keys(): #tag
		tagwise_average_frequency[k2] = tagwise_average_frequency.get(k2,0)+v[k2]

for k in tagwise_average_frequency:
	tagwise_average_frequency[k]/=3

tagwise_average_percentage={}
for k in percentage_distribution_across_folds.keys():
	v = percentage_distribution_across_folds[k]
	for k2 in v.keys(): #tag
		tagwise_average_percentage[k2] = tagwise_average_percentage.get(k2,0)+v[k2]

for k in tagwise_average_percentage:
	tagwise_average_percentage[k]/=3

A=0
B=0
for k in ambiguous_matrix_across_folds.keys():
	a,b = ambiguous_matrix_across_folds[k]
	A+=a
	B+=b
A/=3
B/=3

print("Average scores across the 3 folds")
print("Tag-wise")
print(tagwise_average_scores)
print("Overall")
print(average_scores)
print("Average frequency distribution of tags across folds")
print(tagwise_average_frequency)
print("Average percentage of words that are ambiguous: ",A)
print("Average percentags occurence of such ambiguous words: ",B)


#===========================================================
#***********************************************************
#===========================================================
#					CROSS - VALIDATION STARTS HERE
#							MARKOV LENGTH 2
#===========================================================
#***********************************************************
#===========================================================

# PRE-PROCESSING 
new=[]
start=["^_^"]
end=["._."]
for i in processed_lines:
  new.append(start+i+end)

# PARTIONING
length = len(new)
mod = length%3
part = int((length-mod)/3)
part1 = new[:part+mod]
part2 = new[part+mod:(2*part)+mod]
part3 = new[(2*part)+mod:]
parts=[part1,part2,part3]

matrices_across_folds = {}
overall_matrices_across_folds = {}
frequnecy_distribution_across_folds = {}
percentage_distribution_across_folds = {}
ambiguous_matrix_across_folds = {}

for phase in range(3): # 0,1,2
	train = parts[(phase)%3] + parts[(phase+1)%3]
	test = parts[(phase+2)%3]

	# PRE-PROCESSING OF TOKENS
	tag_set = set([])  # all the tags from train data
	word_set = set([]) # all words from train data
	for i in train:
		tokens = i
		for j in tokens:
			word_tag = j.split("_")
			tag_set.add(word_tag[1])
			word_set.add(word_tag[0]) 

	tags = list(tag_set)
	template_tags = []
	tags_index = {}
	l = len(tags)
	for i in range(l):
		template_tags.append(0)
		tags_index[tags[i]] = i

	words = list(word_set)
	template_words = []
	words_index = {}
	l = len(words)
	for i in range(l):
		template_words.append(0)
		words_index[words[i]] = i

	# FINDING EMISSION AND TRANSITION COUNTS
	emission_counts = {} # tag : [countS corresponding to all words]
	total_counts_for_all_tags = {} 

	for i in tags:
		emission_counts[i] = template_words[:]

	total_transitions = 0 
	transition_counts = {} # {state1:  {state2 : count }}

	ambiguity={}
	occurences={}

	for i in tags:
		temp = {}
		for j in tags:
			temp[j]=0
		transition_counts[i]=temp

	for i in train:
	  tokens = i
	  for j in range(len(tokens)-1):
	    token = tokens[j]
	    word_tag = token.split("_")
	    s1 = word_tag[1]
	    token = tokens[j+1]
	    word_tag = token.split("_")
	    s2 = word_tag[1]
	    total_trans += 1
	    transition_counts[s1][s2] += 1

	for i in train:
		tokens = i
		for j in tokens:
			word_tag = j.split("_")
			word = word_tag[0]
			tag = word_tag[1]
			index = words_index[word]
			emission_counts[tag][index] += 1
			index = tags_index[tag]
			total_counts_for_all_tags[tag] = total_counts_for_all_tags.get(tag,0)+1
			if word in occurences:
			  occurences[word]+=1
			  ambiguity[word].add(tag)
			else:
			  occurences[word]=1
			  ambiguity[word]=set([tag])


	total_trans = 0 
	trans_counts = {} #{s1: {s2 : {s3:count }}}

	# INITIALIZING
	for t1 in tag_set:
	  a={}
	  for t2 in tag_set:
	    b={}
	    for t3 in tag_set:
	      b[t3] = 0
	    a[t2] = b
	  trans_counts[t1] = a

	# FILLING
	for i in train:
	  tokens = i
	  for j in range(len(tokens)-2):
	    token = tokens[j]
	    word_tag = token.split("_")
	    s1 = word_tag[1]
	    token = tokens[j+1]
	    word_tag = token.split("_")
	    s2 = word_tag[1]
	    token = tokens[j+2]
	    word_tag = token.split("_")
	    s3 = word_tag[1]    
	    total_trans += 1
	    trans_counts[s1][s2][s3] +=1


	#===========================================================
	#					TESTING DATA IS SEEN HERE
	#===========================================================
	new_words = set([]) # have zero occurence counts
	new_tags = set([]) # have zero occurence counts

	# SEPARATING WORDS AND TAGS OF THE TEST DATA
	test_data=[]
	actual_tags = []
	for i in test:
		tokens = i
		words=[]
		tags=[]
		for j in tokens:
			word_tag = j.split("_")
			word = word_tag[0]
			tag = word_tag[1]
			words.append(word)
			tags.append(tag)
			if word not in word_set:
				new_words.add(word)
			if tag not in tag_set:
				new_tags.add(tag)
		test_data.append(words)
		actual_tags.append(tags)

	# FINDING TOTAL SIZES OF VOCABULARIES
	all_words = set(list(word_set)+list(new_words))
	vocab_of_words = len(all_words)
	all_tags = set(list(tag_set)+list(new_tags))
	vocab_of_tags = len(all_tags)
	# transition_size = (len(all_tags))**3

	#===========================================================
	#		CALCULATING PROBABILITIES WITH SMOOTHING
	#===========================================================
	total_tag_counts = sum(list(total_counts_for_all_tags.values()))

	# PROBABILITY OF OCCURENCE OF A TAG
	tag_probabs = {} 
	for k in emission_counts.keys(): # k = tag
	  v = sum(emission_counts[k])
	  tag_probabs[k] = (v+1)/(total_tag_counts + vocab_of_tags)
	for i in new_tags:
	  tag_probabs[i] = 1/(total_tag_counts + vocab_of_tags)

	# PROBABILITY OF A WORD GIVEN A TAG
	word_given_tag_probabs={}  #tag: {word : count}
	for i in tag_set:
	  word_probabs={}
	  for j in word_set:
	    index = words_index[j]
	    word_probabs[j] = (emission_counts[i][index] +1)/(tag_probabs[i]*total_tag_counts + vocab_of_words)
	  for j in new_words:
	    word_probabs[j] = 1/(tag_probabs[i]*total_tag_counts + vocab_of_words)
	  word_given_tag_probabs[i] = word_probabs

	for i in new_tags:
	  word_probabs={}
	  for j in all_words:
	    word_probabs[j] = 1/(tag_probabs[i]*total_tag_counts + vocab_of_words)
	  word_given_tag_probabs[i] = word_probabs


	# PROBABILITY OF A STATE(SAY 2) GIVEN PREVIOUS STATE(SAY 1)
	s2_given_s1_probabs={} #tag2: {tag1 : count}
	for t2 in tag_set:
	  t1_probabs={}
	  for t1 in tag_set:
	    t1_probabs[t1] = (transition_counts[t1][t2] + 1)/(tag_probabs[t1]*total_tag_counts + total_trans)
	  for t1 in new_tags:
	    t1_probabs[t1] = 1/(tag_probabs[t1]*total_tag_counts + total_trans)
	  s2_given_s1_probabs[t2] = t1_probabs

	for t2 in new_tags:
	  t1_probabs={}
	  for t1 in all_tags:
	    t1_probabs[t1] = 1/(tag_probabs[t1]*total_tag_counts + total_trans)
	  s2_given_s1_probabs[t2] = t1_probabs

	  # PROBABILITY OF s3 given (s1,s2) IN LOG SPACE
	s3_given_s2_and_s1_probabs={} #smoothed tag3: {(tag1,tag2) : count}
	for t3 in all_tags:
	  t2_and_t1_probabs={}
	  for t2 in all_tags:
	    # t1_probabs={}
	    for t1 in all_tags:
	      try:
	        c = trans_counts[t1][t2][t3]
	      except KeyError as ke:
	        c=0
	      t2_and_t1_probabs[(t1,t2)] = log10((c + 1)/(s2_given_s1_probabs[t2][t1]*(tag_probabs[t1]*total_tag_counts) + total_trans))
	  s3_given_s2_and_s1_probabs[t3] = t2_and_t1_probabs

	#===========================================================
	#				CALCULATING STATS
	#===========================================================

	for tag in new_tags:
	  total_counts_for_all_tags[tag] = 0 #frequencies

	print("Statistics of tags: frequency distribution:")
	print(total_counts_for_all_tags)
	frequnecy_distribution_across_folds[phase] = total_counts_for_all_tags

	percentages={}
	for i in all_tags:
	  percentages[i] = (total_counts_for_all_tags[i] * 100)/total_tag_counts
	print("Statistics of tags: percentage distribution:")
	print(percentages)
	percentage_distribution_across_folds[phase] = percentages


	ambiguous={}
	for k in ambiguity.keys():
	  if len(ambiguity[k])>1:
	    ambiguous[k]=len(ambiguity[k])

	total_occurences = sum(list(occurences.values()))
	ambiguous_occurences={}
	for k in ambiguous.keys():
	  ambiguous_occurences[(k,ambiguous[k])] = (100*occurences[k])/total_occurences
	# print("Words | Tags | Percentage of occurences")
	# print(ambiguous_occurences)
	a=(len(ambiguous)*100)/len(all_words)
	b=sum(list(ambiguous_occurences.values()))
	print("Percentage of words that are ambiguous")
	print(a)
	print("Total occurences percentage of ambiguous words")
	print(b)
	ambiguous_matrix_across_folds[phase] = [a,b]

	#===========================================================
	#				TESTING
	#===========================================================
	results=[]

	for i in test_data:
	  result=[]
	  tokens = i
	  entry_tag = tokens[1]
	  probabilities1 = {}
	  for t in all_tags:
	    probabilities1["^ ^"] = log10(1)

	  #P(s|o)
	  for j in range(len(tokens)):
	    old_word = tokens[j]
	    #p(o1|s1)
	    probab_word_given_tags={}
	    for t in all_tags:
	      probab_word_given_tags[t] = log10(word_given_tag_probabs[t][old_word])

	    #p(s3|s2 and s1) 
	    #s3_given_s2_and_s1_probabs

	    #GOAL: p(o1|s1)*p(older(till s1)) = P
	    P = {} # s1 : p(o1|s1)*p(older(till s1))
	    for k in probabilities1.keys(): 
	      tags = k.split(" ")
	      last_tag = tags[-2] # gives s1
	      P[last_tag] = probabilities1[k] + probab_word_given_tags[last_tag] # both already in log space
	      
	    next_probabilities={}
	    for t3 in all_tags:
	      probabs = {}
	      for k in probabilities1.keys():
	        v = probabilities1[k]
	        l = k.split(" ")
	        t1 = l[-2]
	        t2 = l[-1]
	        probabs[k] = P[t1] + s3_given_s2_and_s1_probabs[t3][(t1,t2)] # already in log space

	      key_max = max(probabs, key=probabs.get) # key_max = t1 
	      # key_min = min(probabs, key=probabs.get) # key_max = t1 
	      # print(probabs[key_min])
	      value_max = probabs[key_max]
	      l = key_max.split(" ")
	      l.append(t3)
	      v = value_max 
	      k = " ".join(i for i in l)
	      next_probabilities[k]=v
	    probabilities1 = next_probabilities
	    
	  key_max = max(probabilities1, key=probabilities1.get) # best tag set
	  result = key_max
	  results.append(result.split(" ")[:-2])
	print(results)
	#===========================================================
	#				CALCULATING MATRICES
	#===========================================================

	confusion_matrix = np.zeros((dimension,dimension))

	for i in range(len(actual_tags)):
	  for j in range(len(actual_tags[i])):
	    at = actual_tags[i][j]
	    pt = results[i][j]
	    ati = all_tags_index[at]
	    pti = all_tags_index[pt]
	    confusion_matrix[ati][pti]+=1

	total = np.sum(confusion_matrix)
	predicted_totals = np.sum(confusion_matrix, axis = 0)
	actual_totals = np.sum(confusion_matrix, axis = 1)

	# print(total)


	tag_scores = {} # tag : {TP: count , FP: count, TN: count}
	for tag in all_tags:
	  index = all_tags_index[tag] # tag to an index
	  scores={}
	  tp = confusion_matrix[index][index]
	  scores["TP"] = tp
	  fp = predicted_totals[index] - tp
	  scores["FP"] = fp
	  fn = actual_totals[index] - tp
	  scores["FN"] = fn
	  # tn = total2 - actual_totals[index] - predicted_totals[index] + tp
	  # scores["TN"] = tn
	  tag_scores[tag] = scores

	matrices={}
	for tag in all_tags:
	  c = tag_scores[tag]
	  precision = c["TP"]/(c["TP"]+c["FP"])
	  precision = 0 if math.isnan(precision) else precision
	  recall = c["TP"]/(c["TP"]+c["FN"])
	  recall = 0 if math.isnan(recall) else recall
	  temp = [precision,recall]
	  f1_score = hm(temp)
	  matrices[tag] = [precision,recall, f1_score]
	  matrices_across_folds[tag] = {phase:[precision,recall,f1_score]}


	print("Tag-wise precision, recall and f1-score")
	print(matrices)

	overall_matrices=[]
	all_precisions = []
	all_recalls=[]
	all_f1_scores=[]
	for k in matrices.keys():
	  a = matrices[k]
	  all_precisions.append(a[0])
	  all_recalls.append(a[1])
	  all_f1_scores.append(a[2])

	overall_matrices.append(get_average(all_precisions))
	overall_matrices.append(get_average(all_recalls))
	overall_matrices.append(get_average(all_f1_scores))

	overall_matrices_across_folds[phase] = overall_matrices

	print("Precision, recall and f1-score")
	print(overall_matrices)


#===========================================================
#					3 FOLDS END HERE
#===========================================================

tagwise_average_scores = {}
for k in matrices_across_folds.keys(): # k = tag
	phase_wise = list(matrices_across_folds[k].values()) #[[p1,r1,f1],[p2,r2,f2],[p3,r3,f3]]
	avg_pre = 0
	avg_rec = 0
	avg_f1 = 0
	for i in phase_wise:
		avg_pre+=i[0]
		avg_rec+=i[1]
		avg_f1+=i[2]
	avg_pre/=3
	avg_rec/=3
	avg_f1/=3
	tagwise_average_scores[k] = [avg_pre,avg_rec,avg_f1]

average_scores = []
phase_wise = list(overall_matrices_across_folds.values()) #[[p1,r1,f1],[p2,r2,f2],[p3,r3,f3]]
avg_pre = 0
avg_rec = 0
avg_f1 = 0
for i in phase_wise:
	avg_pre+=i[0]
	avg_rec+=i[1]
	avg_f1+=i[2]
avg_pre/=3
avg_rec/=3
avg_f1/=3
average_scores = [avg_pre,avg_rec,avg_f1]

tagwise_average_frequency={}
for k in frequnecy_distribution_across_folds.keys():
	v = frequnecy_distribution_across_folds[k]
	for k2 in v.keys(): #tag
		tagwise_average_frequency[k2] = tagwise_average_frequency.get(k2,0)+v[k2]

for k in tagwise_average_frequency:
	tagwise_average_frequency[k]/=3

tagwise_average_percentage={}
for k in percentage_distribution_across_folds.keys():
	v = percentage_distribution_across_folds[k]
	for k2 in v.keys(): #tag
		tagwise_average_percentage[k2] = tagwise_average_percentage.get(k2,0)+v[k2]

for k in tagwise_average_percentage:
	tagwise_average_percentage[k]/=3

A=0
B=0
for k in ambiguous_matrix_across_folds.keys():
	a,b = ambiguous_matrix_across_folds[k]
	A+=a
	B+=b
A/=3
B/=3

print("Average scores across the 3 folds")
print("Tag-wise")
print(tagwise_average_scores)
print("Overall")
print(average_scores)
print("Average frequency distribution of tags across folds")
print(tagwise_average_frequency)
print("Average percentage of words that are ambiguous: ",A)
print("Average percentags occurence of such ambiguous words: ",B)


