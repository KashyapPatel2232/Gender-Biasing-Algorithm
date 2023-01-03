import spacy
from collections import Counter, defaultdict
import numpy as np
# detect gender => male or female
import gender_guesser.detector as gender
d = gender.Detector()
import nltk
import os
import nlp
import random
def Punctuation(string):
 
    # punctuation marks
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, " ")
 
    # Print string without punctuation
    return string

nlp = spacy.load('en_core_web_sm')
file_name = 'D:\\Python\\Data sets\\Open IIT Data Anaytics Competition\\text-and-id.txt'
introduction_file_text = open(file_name).read()
introduction_file_doc = nlp(introduction_file_text)
# Extract tokens for the given doc
#print ([token.text for token in introduction_file_doc])

############################## test code for sentence #######################################
# make array of sentences
# stc = open("D:\\Python\\Data sets\\Open IIT Data Anaytics Competition\\text-and-id.txt",'r')
# stc_arr = nlp(stc)
# # make array of words
# array = [token.text for token in stc_arr]
# print(array)
#############################################################################################

# words and qualifiers which helps to finds the relationship of object and subject 

lemma_female = ["her","she","girl", "woman", "female","daughter"]
lemma_male = ["him","he","boy", "man", "male","son"]
plural_form_male = ['men','boys', 'males']
plural_form_female = ['women','girls','females']
necessity_qualifiers = ['must', 'should', 'ought', 'required', 'have to', 'has to']
quantity_qualifiers = ['all', 'only', 'most']
gend_det_opt = ["unknown","male","andy","male","female","mostly_male","mostly_female"]

# 2D array of text file with each line as a row of array
id_text_array = []
with open(file_name,'r') as f:
    for line in f.readlines():
        line = Punctuation(line)
        id_text_array.append(line.split("\n"))
#print(np.size(id_text_array))
#print(id_text_array[1999][0])
#print('\n')

# find the root words of the sentence to convert the whole documents words into its root form
# Remove stop words and punctuation symbols
# 2D root words array with each row as an array with first element as id 
id_rt_wd_array = []
for i in range(2000):
    txt = nlp(id_text_array[i][0])
    root_words = [token.text for token in txt
            if not token.is_stop and not token.is_punct]
    a = np.array(root_words)
    id_rt_wd_array.append(a)

#print(id_rt_wd_array[0][:])

# find subject from the sentence 
def get_subject_phrase(doc):
    list_sub = []
    for token in doc:
        if ((token.dep_ == "subj") or (token.text.lower() in plural_form_male) or (token.text.lower() in plural_form_female) or (token.text.lower() in lemma_female) or (token.text.lower() in lemma_male)):
            list_sub.append(token.text)
        if token.ent_type_ == 'PERSON':
            gender = d.get_gender(token.text)
            if gender in ["male","female"]:
               list_sub.append(gender)
    return list_sub

# find object from the sentence            
def get_object_phrase(doc):
    list_obj = []
    for token in doc:
        if ("dobj" in token.dep_):
            list_obj.append(token.text)
    return list_obj

#########################    subject array and object array with thier id     #################################################
id_sub = []
id_obj = []

for x in range(2000):
    text = " ".join(id_rt_wd_array[x][1:])
    text = text.lower()
    #print(text)
    text_nlp = nlp(str(text))
    #print(text_nlp)
    subjects = get_subject_phrase(text_nlp)
    #print(subjects)
    objects  = get_object_phrase(text_nlp)
    #print(objects)
    id_arr1 = [id_rt_wd_array[x][0]]
    #print(id_arr1)
    id_arr2 = [id_rt_wd_array[x][0]]
    #print(id_arr2)
    id_arr1=np.array(id_arr1)
    #print(type(id_arr1))
    id_arr2= np.array(id_arr2)
    subject = np.array(subjects)
    object = np.array(objects)
    id_sub.append(np.array(np.concatenate((id_arr1,subject))))
    id_obj.append(np.array(np.concatenate((id_arr2,object))))

#print(id_sub[0])
#print(id_obj[0])
############################################################################################################################

##################################### qualifiers not implementing right now !!! #####################################################
# def Qualifier_check(sentc):
#     for word in sentc:
#         if ( (word in necessity_qualifiers) or ( word in quantity_qualifiers) ):
#             return True
#     return False

# for sentence in stc:
#     #if Qualifier_check(sentence):
#         doc = nlp(sentence)
#         subject_phrase = get_subject_phrase(doc)
#         object_phrase = get_object_phrase(doc)
#         print("Subject :")
#         print(subject_phrase)
#         print("\n")
#         print("Object :")
#         print(object_phrase)
#         print("\n")

#######################################################################################################

################################# algorithm for finding the cosine similarity between the words using CBOW(Continuous bag of words) and Gram Skip  #######################################
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
 
warnings.filterwarnings(action = 'ignore')
 
import gensim
from gensim.models import Word2Vec
 
#  Reads ‘text-and-id.txt’ file
sample = open("D:\\Python\\Data sets\\Open IIT Data Anaytics Competition\\text-and-id.txt")
s = sample.read()
 
# Replaces escape character with space
f = Punctuation(s)
k = f.replace("\n", " ")
 
data = []
 
# iterate through each sentence in the file
for i in sent_tokenize(k):
    temp = []
     
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
 
    data.append(temp)
 
# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1)
 
# CBOW model
#CBOW = model1.wv.similarity('men', 'society')
 
# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, sg = 1)
 
# Skip Gram
#SG = model2.wv.similarity('men', 'society')
     


#####################################################################################################     

# CBOW = model1.wv.similarity(id_sub[0][1],id_obj[0][1])
# SG = model2.wv.similarity(id_sub[0][1],id_obj[0][1])
# CBOW2 = model1.wv.similarity('boys',id_obj[0][1])
# SG2 = model2.wv.similarity('boys',id_obj[0][1])

# Similarity = (CBOW + SG)/2
# Similarity2 = (CBOW2 + SG2)/2
# print(Similarity)
# print(Similarity2)
# if abs((Similarity-Similarity2) > 0.02):
#     print("Biased")
# else :
#     print("Unbiased")

# form dictionary if sentence is biased or unbiased with key as thier id
########################### CBOW + Skip Gram model ##########################################
Dict = {}
for i in range(2000):
    num_female_lemma = 0
    num_male_lemma = 0
    # if no subject present in sentencertc        98ewq  
    if len(id_sub[i]) == 1:
        Dict[str(id_sub[i][0])] = 'u'

    # if no object is present in sentence
    elif len(id_obj[i]) == 1:
        for j in id_sub[i]:
            if (j in lemma_male) or (j in plural_form_male):
                num_male_lemma += 1

            if (j in lemma_female) or (j in plural_form_female): 
                num_female_lemma += 1 
        
        if num_male_lemma == num_female_lemma :
            Dict[str(id_sub[i][0])] =  random.choice(['b','u'])

        else:
            Dict[str(id_sub[i][0])] = 'b'

    else:
        for k in id_sub[i]:
            if (k in lemma_male) :
                idx = lemma_male.index(k)
                oppo_gen = lemma_female[idx]
                for alpha in id_obj[i]:
                    CBOW_male = model1.wv.similarity(k,alpha)
                    SG_male = model2.wv.similarity(k,alpha)
                    CBOW_female = model1.wv.similarity(oppo_gen,alpha)
                    SG_female = model2.wv.similarity(oppo_gen,alpha)

                    Similarity_male = (CBOW_male + SG_male)/2
                    Similarity_female = (CBOW_female + SG_female)/2
                    if (abs(Similarity_male-Similarity_female) > 0.01):
                        Dict[id_sub[i][0]] = 'b'
                    else :
                        Dict[id_sub[i][0]] = 'u'

            elif (k in plural_form_male):
                idx = plural_form_male.index(k)
                oppo_gen = plural_form_female[idx]
                for alpha in id_obj[i]:
                    CBOW_male = model1.wv.similarity(k,alpha)
                    SG_male = model2.wv.similarity(k,alpha)
                    CBOW_female = model1.wv.similarity(oppo_gen,alpha)
                    SG_female = model2.wv.similarity(oppo_gen,alpha)

                    Similarity_male = (CBOW_male + SG_male)/2
                    Similarity_female = (CBOW_female + SG_female)/2
                    if (abs(Similarity_male-Similarity_female) > 0.01):
                        Dict[id_sub[i][0]] = 'b'
                    else :
                        Dict[id_sub[i][0]] = 'u'
            elif (k in lemma_female) :
                idx = lemma_female.index(k)
                oppo_gen = lemma_male[idx]
                for alpha in id_obj[i]:
                    CBOW_female = model1.wv.similarity(k,alpha)
                    SG_female = model2.wv.similarity(k,alpha)
                    CBOW_male = model1.wv.similarity(oppo_gen,alpha)
                    SG_male = model2.wv.similarity(oppo_gen,alpha)

                    Similarity_male = (CBOW_male + SG_male)/2
                    Similarity_female = (CBOW_female + SG_female)/2
                    if (abs(Similarity_male-Similarity_female) > 0.01):
                        Dict[id_sub[i][0]] = 'b'
                    else :
                        Dict[id_sub[i][0]] = 'u'

            elif (k in plural_form_female):
                idx = plural_form_female.index(k)
                oppo_gen = plural_form_male[idx]
                for alpha in id_obj[i]:
                    CBOW_female = model1.wv.similarity(k,alpha)
                    SG_female = model2.wv.similarity(k,alpha)
                    CBOW_male = model1.wv.similarity(oppo_gen,alpha)
                    SG_male = model2.wv.similarity(oppo_gen,alpha)

                    Similarity_male = (CBOW_male + SG_male)/2
                    Similarity_female = (CBOW_female + SG_female)/2
                    if (abs(Similarity_male-Similarity_female) > 0.01):
                        Dict[id_sub[i][0]] = 'b'
                    else :
                        Dict[id_sub[i][0]] = 'u'
            
            else:
                Dict[id_sub[i][0]] = 'u'

#print(Dict)

######################################################################################

################################ CBOW Model#######################################################
# Dict = {}
# for i in range(2000):
#     num_female_lemma = 0
#     num_male_lemma = 0
#     # if no subject present in sentence
#     if len(id_sub[i]) == 1:
#         Dict[str(id_sub[i][0])] = 'u'

#     # if no object is present in sentence
#     elif len(id_obj[i]) == 1:
#         for j in id_sub[i]:
#             if (j in lemma_male) or (j in plural_form_male):
#                 num_male_lemma += 1

#             if (j in lemma_female) or (j in plural_form_female): 
#                 num_female_lemma += 1 
        
#         if num_male_lemma == num_female_lemma :
#             Dict[str(id_sub[i][0])] =  random.choice(['b','u'])

#         else:
#             Dict[str(id_sub[i][0])] = 'b'

#     else:
#         for k in id_sub[i]:
#             if (k in lemma_male) :
#                 idx = lemma_male.index(k)
#                 oppo_gen = lemma_female[idx]
#                 for alpha in id_obj[i]:
#                     CBOW_male = model1.wv.similarity(k,alpha)
#                     CBOW_female = model1.wv.similarity(oppo_gen,alpha)

#                     Similarity_male = CBOW_male 
#                     Similarity_female = CBOW_female 
#                     if (abs(Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'

#             elif (k in plural_form_male):
#                 idx = plural_form_male.index(k)
#                 oppo_gen = plural_form_female[idx]
#                 for alpha in id_obj[i]:
#                     CBOW_male = model1.wv.similarity(k,alpha)
#                     CBOW_female = model1.wv.similarity(oppo_gen,alpha)

#                     Similarity_male = CBOW_male
#                     Similarity_female = CBOW_female 
#                     if (abs(Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'
#             elif (k in lemma_female) :
#                 idx = lemma_female.index(k)
#                 oppo_gen = lemma_male[idx]
#                 for alpha in id_obj[i]:
#                     CBOW_female = model1.wv.similarity(k,alpha)
#                     CBOW_male = model1.wv.similarity(oppo_gen,alpha)

#                     Similarity_male = CBOW_male 
#                     Similarity_female = CBOW_female
#                     if (abs(Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'

#             elif (k in plural_form_female):
#                 idx = plural_form_female.index(k)
#                 oppo_gen = plural_form_male[idx]
#                 for alpha in id_obj[i]:
#                     CBOW_female = model1.wv.similarity(k,alpha)
#                     CBOW_male = model1.wv.similarity(oppo_gen,alpha)

#                     Similarity_male = CBOW_male
#                     Similarity_female = CBOW_female
#                     if (abs(Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'
            
#             else:
#                 Dict[id_sub[i][0]] = 'u'

# print(Dict)

###############################################################################################

################################ Skip Gram Model #######################################################
# Dict = {}
# for i in range(2000):
#     num_female_lemma = 0
#     num_male_lemma = 0
#     # if no subject present in sentence
#     if len(id_sub[i]) == 1:
#         Dict[str(id_sub[i][0])] = 'u'

#     # if no object is present in sentence
#     elif len(id_obj[i]) == 1:
#         for j in id_sub[i]:
#             if (j in lemma_male) or (j in plural_form_male):
#                 num_male_lemma += 1

#             if (j in lemma_female) or (j in plural_form_female): 
#                 num_female_lemma += 1 
        
#         if num_male_lemma == num_female_lemma :
#             Dict[str(id_sub[i][0])] =  random.choice(['b','u'])

#         else:
#             Dict[str(id_sub[i][0])] = 'b'

#     else:
#         for k in id_sub[i]:
#             if (k in lemma_male) :
#                 idx = lemma_male.index(k)
#                 oppo_gen = lemma_female[idx]
#                 for alpha in id_obj[i]:
#                     SG_male = model2.wv.similarity(k,alpha)
#                     SG_female = model2.wv.similarity(oppo_gen,alpha)

#                     Similarity_male =  SG_male
#                     Similarity_female = SG_female
#                     if (abs(Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'

#             elif (k in plural_form_male):
#                 idx = plural_form_male.index(k)
#                 oppo_gen = plural_form_female[idx]
#                 for alpha in id_obj[i]:
#                     SG_male = model2.wv.similarity(k,alpha)
#                     SG_female = model2.wv.similarity(oppo_gen,alpha)

#                     Similarity_male = SG_male
#                     Similarity_female = SG_female
#                     if (abs(Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'
#             elif (k in lemma_female) :
#                 idx = lemma_female.index(k)
#                 oppo_gen = lemma_male[idx]
#                 for alpha in id_obj[i]:
#                     CBOW_female = model1.wv.similarity(k,alpha)
#                     SG_female = model2.wv.similarity(k,alpha)
#                     CBOW_male = model1.wv.similarity(oppo_gen,alpha)
#                     SG_male = model2.wv.similarity(oppo_gen,alpha)

#                     Similarity_male = SG_male
#                     Similarity_female = SG_female
#                     if abs((Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'

#             elif (k in plural_form_female):
#                 idx = plural_form_female.index(k)
#                 oppo_gen = plural_form_male[idx]
#                 for alpha in id_obj[i]:
#                     CBOW_female = model1.wv.similarity(k,alpha)
#                     SG_female = model2.wv.similarity(k,alpha)
#                     CBOW_male = model1.wv.similarity(oppo_gen,alpha)
#                     SG_male = model2.wv.similarity(oppo_gen,alpha)

#                     Similarity_male = SG_male
#                     Similarity_female = SG_female
#                     if (abs(Similarity_male-Similarity_female) > 0.01):
#                         Dict[id_sub[i][0]] = 'b'
#                     else :
#                         Dict[id_sub[i][0]] = 'u'
            
#             else:
#                 Dict[id_sub[i][0]] = 'u'

#print(Dict)

##################################################################################################################

# pair-label 2D array
pairs_label = []
file_name2 = "D:\Python\Data sets\Open IIT Data Anaytics Competition\pairs-label-training (1).txt"
with open(file_name2,'r') as f:
    for line in f.readlines():
        line_array= line.split("\n")
        pairs_label.append(line_array[0].split(', '))

#print(pairs_label[-1][:])

############################### % accuracy of model ####################
correct_predictions = 0
# total = 155951
total = 155951
for num in range (total):
    # print(pairs_label[num][0])
    # print(pairs_label[num][1])
    # print(Dict[pairs_label[num][0]])
    # print(Dict[pairs_label[num][1]])
    # print(type(pairs_label[num][2]))
    # print("\n")

    if (Dict[pairs_label[num][0]] == 'b' and Dict[pairs_label[num][1]] == 'b' and pairs_label[num][2] == '0'):
        correct_predictions += 1

    elif (Dict[pairs_label[num][0]] == 'u' and Dict[pairs_label[num][1]] == 'u' and pairs_label[num][2] == '0'):
        correct_predictions += 1
        
    elif(Dict[pairs_label[num][0]] == 'u' and Dict[pairs_label[num][1]] == 'b' and pairs_label[num][2] == '1'):
        correct_predictions += 1

    elif(Dict[pairs_label[num][0]] == 'b' and Dict[pairs_label[num][1]] == 'u' and pairs_label[num][2] == '1'):
        correct_predictions += 1

model_accuracy = (correct_predictions/total)*100
print("CBOW + skip gram model accuracy:")
print(model_accuracy)

