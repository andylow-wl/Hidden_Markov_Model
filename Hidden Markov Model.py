#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Members:
#1. Andy Low Wei Liang         
#2. 
#3. 
#4.

# Codes not used in the run() function has been commented out
# ddir required to be inserted to use working directory

import random
import pandas as pd
import numpy as np
import re

## Obtain list of possible tags
def obtain_tags(tags_file):
    tags_f = open(tags_file, encoding = "utf-8")
    tags_ls = tags_f.readlines()
    temp = []
    for t in tags_ls:
        temp.append(t.rstrip('\n'))
    return temp
def read_tweets(in_train_filename):
    f = open(in_train_filename, encoding="utf8")
    lines = f.readlines()
    tweets = []
    temp = []
    for l in lines:
        l = l.rstrip('\n')
        temp.append(l)
        if l == '':
            tweets.append(temp)
            temp = []
    return tweets
def rand_tags(tags_ls, tweets, seed):
    tags = []
    random.seed(seed)
    for t in tweets:
        temp = []
        for i in range(len(t)):
            if i == len(t) - 1:
                tags.append(temp)
                temp = []
            else:
                temp.append(tags_ls[random.randint(0,len(tags_ls)-1)])
    return tags
def rand_emiss_trans(tags_ls, tweets, tags, sigma=0.01):
    word = []
    for t in tweets:
        for w in t:
            word.append(w)
    word = list(set(word)) # Remove duplicates
    word.remove('') # Obtained rows index for emiss_df

    ## Initiate emiss_df with all zero
    emiss_df = np.zeros(shape=(len(word),len(tags_ls)))
    emiss_df = pd.DataFrame(emiss_df, columns=tags_ls)
    emiss_df['index'] = word
    emiss_df = emiss_df.set_index('index') # Rename index as tags

    ## Initiate trans_df with all zero
    trans_df = np.zeros(shape=(len(tags_ls)+1,len(tags_ls)+1))
    tags_ls2 = tags_ls.copy()
    tags_ls2.append('STOP')
    trans_df = pd.DataFrame(trans_df, columns=tags_ls2)
    tags_ls3 = tags_ls.copy()
    tags_ls3.append('START')
    trans_df['index'] = tags_ls3
    trans_df = trans_df.set_index('index') # Rename index as tags

    ## Updating trans_df & emiss_df with tweets and randomised tags count
    length = len(tweets) # Length of tweets == Length of tags
    for i in range(length):
        tweet = tweets[i]
        tag = tags[i]
        length2 = len(tweet) # Length of tweet-1 == Length of tag
        for j in range(length2):
            if tweet[j] == '':
                trans_df.at[tag[j-1],'STOP'] = trans_df.at[tag[j-1],'STOP'] + 1 # Updating of trans_df stop
            else:
                emiss_df.at[tweet[j],tag[j]] = emiss_df.at[tweet[j],tag[j]] + 1 # Updating of emiss_df
                if j == 0:
                    trans_df.at['START',tag[j]] = trans_df.at['START',tag[j]] + 1 # Updating of trans_df start
                else:
                    trans_df.at[tag[j-1],tag[j]] = trans_df.at[tag[j-1],tag[j]] + 1 # Updating of trans_df

    ## Processing trans_df counts to probability
    sum_ls = []
    for i in range(len(tags_ls)+1):
        sum_ls.append(trans_df.iloc[i,].values.sum())
    trans_df['sum'] = sum_ls

    for i in trans_df.columns:
        trans_df[i] = (trans_df[i] + sigma) / (trans_df['sum'] + ((len(tags_ls) + 2) * sigma))
    trans_df = trans_df.drop('sum',axis=1)

    ## Processing emiss_df counts to probability
    ## Creating dictionary to count total tags
    tag_dict = dict.fromkeys(tags_ls)
    for tag in tags_ls:
        tag_dict[tag] = emiss_df[tag].values.sum()

    for w in word:
        for tag in emiss_df.columns:
            emiss_df.at[w,tag] = (emiss_df.at[w,tag] + sigma) / (tag_dict[tag] + (sigma * (len(word) +1)))

    return emiss_df, trans_df
def compute_alpha(tweet, tags_ls, trans_df, emiss_df):
    alpha = []
    counter = 0
    for word in tweet:
        alpha_dict = dict.fromkeys(tags_ls,'')
        if word == '':
            alpha_before = alpha[-1]
            ls = []
            for tag in tags_ls:
                ls.append(alpha_before[tag] * trans_df.at[tag,'STOP'])
            alpha_dict = sum(ls)
        elif counter == 0 :
            for tag in tags_ls:
                alpha_dict[tag] = trans_df.at['START',tag] * emiss_df.at[word,tag]
        else:
            alpha_before = alpha[-1]
            for tag_j in tags_ls:
                ls = []
                for tag_i in tags_ls:
                    ls.append(alpha_before[tag_i] * trans_df.at[tag_i,tag_j] * emiss_df.at[word,tag_j])
                alpha_dict[tag_j] = sum(ls)
        alpha.append(alpha_dict)
        counter = counter + 1
    return alpha
def compute_beta(tweet, tags_ls, trans_df, emiss_df):
    beta = []
    counter = 0
    for word in reversed(tweet):
        beta_dict = dict.fromkeys(tags_ls,'')

        if word == '':
            for tag in tags_ls:
                beta_dict[tag] = trans_df.at[tag,'STOP']
        elif word == tweet[0] and counter == len(tweet)-1:
            beta_after = beta[0]
            ls = []
            for tag in tags_ls:
                ls.append(beta_after[tag] * trans_df.at['START',tag] * emiss_df.at[word,tag])
            beta_dict = sum(ls)
        else:
            beta_after = beta[0]
            for tag_i in tags_ls:
                ls = []
                for tag_j in tags_ls:
                    ls.append(beta_after[tag_j] * trans_df.at[tag_i,tag_j] * emiss_df.at[word,tag_j])
                beta_dict[tag_i] = sum(ls)
        beta.insert(0,beta_dict)
        counter = counter + 1
    return beta

def compute_xi(tweet, tags_ls, trans_df, emiss_df, alpha, beta):
    denom = alpha[-1]

    ## Create Xi Dataframe since we are summing everything anyways
    xi = np.zeros(shape=(len(tags_ls)+1,len(tags_ls)+1))
    tags_ls2 = tags_ls.copy()
    tags_ls2.append('STOP')
    xi = pd.DataFrame(xi, columns=tags_ls2)
    tags_ls3 = tags_ls.copy()
    tags_ls3.append('START')
    xi['index'] = tags_ls3
    xi = xi.set_index('index') # Rename index as tags

    ## Update Xi dataframe by summing through words
    tweet_len = len(tweet)
    for i in range(tweet_len-2):
        alpha_state = alpha[i]
        beta_state = beta[i+2]
        for tag_i in tags_ls:
            for tag_j in tags_ls:
                xi.at[tag_i,tag_j] = xi.at[tag_i,tag_j] + (alpha_state[tag_i] * trans_df.at[tag_i,tag_j] * emiss_df.at[tweet[i+1],tag_j] * beta_state[tag_j] / denom)
    for tag in tags_ls:
        xi.at['START',tag] = trans_df.at['START',tag] * emiss_df.at[tweet[0],tag] * beta[1][tag] / denom
        xi.at[tag,'STOP'] = alpha[-2][tag] * trans_df.at[tag,'STOP'] / denom

    return xi

def compute_gamma(tweet, tags_ls, alpha, beta):
    tweet_len = len(tweet)
    word = []
    gamma = []
    denom = alpha[-1]
    log = 0
    for i in range(tweet_len-1):
        temp_gamma = []
        alpha_state = alpha[i]
        beta_state = beta[i + 1]
        word.append(tweet[i])
        for tag in tags_ls:
            numer = alpha_state[tag] * beta_state[tag]
            temp_gamma.append(numer / denom)
            if (i == tweet_len-2):
                log = log + numer
        gamma.append(temp_gamma)
    gamma =pd.DataFrame(gamma,columns=tags_ls)
    gamma['index'] = word
    gamma = gamma.set_index('index') # Rename index as word
    log = np.log10(log)
    return gamma,log

def create_xi_frame(tags_ls):
    ## Create Xi Dataframe since we are summing everything anyways
    xi = np.zeros(shape=(len(tags_ls)+1,len(tags_ls)+1))
    tags_ls2 = tags_ls.copy()
    tags_ls2.append('STOP')
    xi = pd.DataFrame(xi, columns=tags_ls2)
    tags_ls3 = tags_ls.copy()
    tags_ls3.append('START')
    xi['index'] = tags_ls3
    xi = xi.set_index('index') # Rename index as tags
    return xi

def create_gamma_frame(tags_ls, tweets):
    word = []
    for t in tweets:
        for w in t:
            word.append(w)
    word = list(set(word)) # Remove duplicates
    word.remove('') # Obtained rows index for emiss_df

    ## Create gamma_frame
    emiss_df = np.zeros(shape=(len(word),len(tags_ls)))
    emiss_df = pd.DataFrame(emiss_df, columns=tags_ls)
    emiss_df['index'] = word
    emiss_df = emiss_df.set_index('index') # Rename index as tags

    return emiss_df

def sum_gamma(tags_ls, total_gamma, gamma):
    word_ls = list(gamma.index.values)
    for word in word_ls:
        for tag in tags_ls:
            total_gamma.at[word,tag] = total_gamma.at[word,tag] + gamma.at[word,tag].sum()
    return total_gamma

def compute_trans(tags_ls, trans_df, xi, sigma = 0.01):
    denom = []
    for i in range(len(tags_ls) + 1):
        denom.append(xi.iloc[i,].values.sum() + ((len(tags_ls) + 2) * sigma))
    xi['denom'] = denom
    for i in trans_df.columns:
        trans_df[i] = (xi[i] + sigma) / (xi['denom'])
    return trans_df

def compute_emiss(tweet, tags_ls, emiss_df, gamma, sigma = 0.01):
    denom = dict.fromkeys(tags_ls)
    for tag in tags_ls:
        denom[tag] = gamma[tag].values.sum()

    ## Updating Emissions Dataframe
    for word in tweet:
        if word == '':
            pass
        else:
            for tag in tags_ls:             
                emiss_df.at[word,tag] = gamma.at[word,tag].sum() / denom[tag]

    ## Normalising the Emissions Dataframe
    sum_ls = []
    for i in range(emiss_df.shape[0]):
        sum_ls.append(emiss_df.iloc[i,].values.sum())
    emiss_df['sum'] = sum_ls

    for tag in tags_ls:
        emiss_df[tag] = emiss_df[tag] / emiss_df['sum']
    emiss_df = emiss_df.drop('sum',axis=1)

    return emiss_df

# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    pass

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    pass

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    pass

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    sigma = 0.01
    ## Creating transitive dataframe
    trans_df = pd.read_csv(in_trans_probs_filename, sep="\t", index_col = 'index')
    trans_df = trans_df.drop(columns="STOP")
    
    ## Creating emissions dataframe
    emiss_df = pd.read_csv(in_output_probs_filename, sep='\t')
    
    ## Changing emission dataframe to lowercase and recalculating mle
    emiss_df['index'] = emiss_df['index'].apply(lambda x: x.lower()) #####
    num_words = emiss_df['index'].nunique() + 1
    emiss_df = emiss_df.set_index('index')
    tags_ls = obtain_tags(in_tags_filename)
    
    ## Calculating Denom for MLE
    denom = dict.fromkeys(tags_ls)
    for tag in tags_ls:
        denom[tag] = emiss_df[tag].values.sum()
    
    ## Updating emiss_df with new MLE
    for row in emiss_df.iterrows():
        word = row[0]
        for tag in tags_ls:
            emiss_df.loc[word,tag] = (emiss_df.loc[word,tag].sum() + sigma) / (denom[tag] + (sigma * num_words))
    
    ## Dropping duplicates
    emiss_df = emiss_df.drop_duplicates()
    
    ## Normalising emiss_df
    sum_ls = []
    for i in range(emiss_df.shape[0]):
        sum_word = emiss_df.iloc[i,].values.sum()
        for j in range(len(tags_ls)):
            emiss_df.iloc[i,j] = emiss_df.iloc[i,j] / sum_word
    
    ## Reading test to a list of tweets(list) of words
    test_df = open(in_test_filename, encoding="utf8")
    test = test_df.readlines()
    tweets = []
    temp = []
    for l in test:
        l = l.rstrip('\n').lower() ##### remove \n from every line and lower
        if l == "":
            tweets.append(temp)
            temp = []
        else:
            temp.append(l)
    
    ## Update denom to mle for unseen words
    for tag in tags_ls:
        denom[tag] = sigma / (denom[tag] + (sigma * num_words))
    
    ## Creating transdf2 for unseen words
    trans_df2 = trans_df.copy()
    for col in tags_ls:
        trans_df2[col] = trans_df2[col].apply(lambda x: x * denom[col])
    
    
    backlog = [] # to record predicted tag
    # Creating dataframe for each word for multiplication
    for t in tweets:
        for word in t:
            # Automatically assign tags for common phrases
            # Could very well be part of preprocessing since we are capturing the same text
            if word == 'rt': # Capture for rt
                backlog.append('~')
            elif word.startswith('@user_'): # Capture for unique users
                backlog.append('@')
            elif word.startswith('http'): # Capture for links
                backlog.append('U')
            elif re.match(r'^(\d+\.)?\d+$',word): # Capture for floats eg. 1.0
                backlog.append('$')
            elif re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$',word): # Capture for thousands 1,000
                backlog.append('$')
            elif re.match(r'/#(\d+)/',word): # Capture for positions e.g. #1
                backlog.append('$')
            elif re.match(r'^#',word): # Capture for hashtags
                backlog.append('#')
            elif word in emiss_df.index.values:
                tag_i = backlog[-1]
                temp_dict = dict.fromkeys(tags_ls)
                for tag_j in tags_ls:
                    temp_dict[tag_j] = emiss_df.loc[word,tag_j] * trans_df.loc[tag_i,tag_j]
                backlog.append(max(temp_dict, key=temp_dict.get) )
            else:
                if word == t[0]:
                    backlog.append(trans_df.loc['START',].idxmax())
                else:
                    tag_subset = trans_df2.loc[backlog[-1],]
                    tag_subset = tag_subset.sort_values(ascending=False)
                    temp_tag = tag_subset.index[0]
                    backlog.append(temp_tag)
    
    with open(out_predictions_filename, "w", encoding="utf8") as outfile:
        outfile.write(str("\n".join(backlog)))

def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):

    ## Read train file and split by tweets
    tweets = read_tweets(in_train_filename)

    ## Read tags file
    tags_ls = obtain_tags(in_tag_filename)
    
    ## Initialise random tags for train file
    tags = rand_tags(tags_ls, tweets, seed)
    
    ## Generate Emissions and Transitions Dataframe from random tags
    sigma = 0.01
    emiss_df, trans_df = rand_emiss_trans(tags_ls, tweets, tags, sigma)
    trans_df.to_csv('trans_probs3.txt', sep='\t')
    emiss_df.to_csv('output_probs3.txt', sep='\t')
    print('Export complete')
    
    ## Creating Xi and Gamma Frame
    thresh_bool = False
    iterations_log = [0]
    
    ## Compute for each iteration
    for iteration in range(max_iter):
        print('Iteration: ' + str(iteration+1))
        local_log = 0
        total_xi = create_xi_frame(tags_ls)
        total_gamma = create_gamma_frame(tags_ls, tweets)
        ## In the iteration that convergence is achieved
        if thresh_bool:
            print('Threshold is reached')
            trans_df.to_csv(out_trans_filename, sep='\t')
            emiss_df.to_csv(out_output_filename, sep='\t')
            break
        ## Compute for each tweet
        for tweet in tweets:
            alpha = compute_alpha(tweet, tags_ls, trans_df, emiss_df)
            beta = compute_beta(tweet, tags_ls, trans_df, emiss_df)
            xi = compute_xi(tweet, tags_ls, trans_df, emiss_df, alpha, beta)
            gamma,log = compute_gamma(tweet, tags_ls, alpha, beta)
            total_xi = total_xi + xi
            total_gamma = sum_gamma(tags_ls, total_gamma, gamma)
            local_log = local_log + log
            if (tweet == tweets[-1]):
                print("Local Log: " + str(local_log))
                iterations_log.append(local_log) 
        trans_df = compute_trans(tags_ls, trans_df, total_xi, sigma)
        emiss_df = compute_emiss(tweet, tags_ls, emiss_df, total_gamma)
        thresh_bool = True if (abs(iterations_log[-1] - iterations_log[-2]) < thresh) else False
        out_name = 'temp_trans'+str(iteration)+'.txt'
        out_name2 = 'temp_out'+str(iteration)+'.txt'
        trans_df.to_csv(out_name, sep='\t')
        emiss_df.to_csv(out_name2, sep='\t')
        
    ## In the event that the threshold is not reached
    if (thresh_bool == False):
        print('Threshold not reached')
        trans_df.to_csv(out_trans_filename, sep='\t')
        emiss_df.to_csv(out_output_filename, sep='\t')
        
    print(iterations_log)
    

def cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                out_predictions_file):
    def read_stocks(in_test_filename):
        f = open(in_test_filename, encoding="utf8")
        lines = f.readlines()
        tweets = []
        temp = []
        for l in lines:
            l = l.rstrip('\n')
            if l == '':
                tweets.append(temp)
                temp = []
            else:
                temp.append(l)
        return tweets
    stocks = read_stocks(in_test_filename)
    emiss_df = pd.read_csv(in_output_probs_filename, sep = '\t', index_col = 'index')
    trans_df = pd.read_csv(in_trans_probs_filename, sep = '\t', index_col = 'index')
    
    states = obtain_tags(in_states_filename)
    
    ## Empty list of predicted change
    prediction = []
    
    ## Prediction for last tag
    outcome_dict = dict.fromkeys(states)
    for state_before in states:
        emiss_df2 = emiss_df.copy()
        for i in range(len(states)):
            for j in range(emiss_df2.shape[0]):
                emiss_df2.iat[j,i] = emiss_df2.iat[j,i] * trans_df.at[state_before,states[i]]
        maximum_ls = []
        for i in range(emiss_df.shape[0]):
            maximum_ls.append(max(emiss_df2.iloc[i,].values))
        emiss_df2['max'] = maximum_ls
        emiss_df2 = emiss_df2.sort_values(by = 'max', ascending=False)
        outcome_dict[state_before] = emiss_df2.iloc[0,].name

    ## Predicting state 
    for stock in stocks:
        counter = 0
        backlog = []
        for change in stock:
            temp_dict = dict.fromkeys(states)
            if counter == 0:
                for state in states:
                    temp_dict[state] = trans_df.at['START',state] * emiss_df.at[int(change), state]
            else:
                state_before = backlog[-1]
                for state in states:
                    temp_dict[state] = trans_df.at[state_before,state] * emiss_df.at[int(change), state]
            backlog.append(max(temp_dict, key=temp_dict.get))
            counter += 1
        
        ## Predicting next change
        prediction.append(str(outcome_dict[backlog[-1]]))
    with open(out_predictions_file, "w", encoding="utf8") as outfile:
        outfile.write(str("\n".join(prediction)))


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth))**2
    return error/len(predicted_tags), error, len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    in_tag_filename     = f'{ddir}/twitter_tags.txt'
    out_trans_filename  = f'{ddir}/trans_probs4.txt'
    out_output_filename = f'{ddir}/output_probs4.txt'
    max_iter = 10
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    trans_probs_filename3 =  f'{ddir}/trans_probs3.txt'
    output_probs_filename3 = f'{ddir}/output_probs3.txt'
    viterbi_predictions_filename3 = f'{ddir}/fb_predictions3.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
                     viterbi_predictions_filename3)
    correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename4 =  f'{ddir}/trans_probs4.txt'
    output_probs_filename4 = f'{ddir}/output_probs4.txt'
    viterbi_predictions_filename4 = f'{ddir}/fb_predictions4.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
                     viterbi_predictions_filename4)
    correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/cat_price_changes_train.txt'
    in_tag_filename     = f'{ddir}/cat_states.txt'
    out_trans_filename  = f'{ddir}/cat_trans_probs.txt'
    out_output_filename = f'{ddir}/cat_output_probs.txt'
    max_iter = 1000000
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    in_test_filename         = f'{ddir}/cat_price_changes_dev.txt'
    in_trans_probs_filename  = f'{ddir}/cat_trans_probs.txt'
    in_output_probs_filename = f'{ddir}/cat_output_probs.txt'
    in_states_filename       = f'{ddir}/cat_states.txt'
    predictions_filename     = f'{ddir}/cat_predictions.txt'
    cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                predictions_filename)

    in_ans_filename     = f'{ddir}/cat_price_changes_dev_ans.txt'
    ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    print(f'average squared error for {num_ex} examples: {ave_sq_err}')

if __name__ == '__main__':
    run()

