"""Utility functions for BERT modeling""" 
import numpy as np
import pandas as pd

import src.data.data_utils as data_utils
import src.utils as utils


def create_bert_extract_features_cmd(df, dataset_name):
    """Create bert extract fetures command for specified dataset"""
    extract_features_cmd = "python {extract_features_script} \
      --input_file={input_file} \
      --output_file={output_file} \
      --vocab_file={vocab_file} \
      --bert_config_file={bert_config_file} \
      --init_checkpoint={init_checkpoint} \
      --layers=-1 \
      --max_seq_length=256 \
      --batch_size=8".format(
        extract_features_script = utils.bert_dir + "extract_features.py",
        input_file = utils.data_interim_dir + "bert_input_{dataset}.txt".format(dataset=dataset_name),
        output_file = utils.data_interim_dir + "bert_output_{dataset}.json".format(dataset=dataset_name),
        vocab_file = utils.models_dir + "uncased_L-12_H-768_A-12/vocab.txt",
        bert_config_file = utils.models_dir + "uncased_L-12_H-768_A-12/bert_config.json",
        init_checkpoint = utils.models_dir + "uncased_L-12_H-768_A-12/bert_model.ckpt"    
    )

    return extract_features_cmd


def read_in_bert_features(dataset_name):
    """Taken from public Kaggle kernal: Taming the BERT - a baseline"""
    bert_output = pd.read_json(
        utils.data_interim_dir + "bert_output_{dataset}.json".format(dataset=dataset_name), lines = True)
    
    return bert_output
    

def create_bert_word_embedding_df(df, bert_output, dataset_name):
    """Taken from public Kaggle kernal: Taming the BERT - a baseline"""
    index = df.index
    columns = ["emb_A", "emb_B", "emb_P", "label"]
    emb = pd.DataFrame(index = index, columns = columns)
    emb.index.name = "ID"

    for i in range(len(df)): # For each line in the data file
        # get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
        P = df.loc[i,"Pronoun"].lower()
        A = df.loc[i,"A"].lower()
        B = df.loc[i,"B"].lower()

        # For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
        P_offset = data_utils.compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"Pronoun-offset"])
        A_offset = data_utils.compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"A-offset"])
        B_offset = data_utils.compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"B-offset"])
        # Figure out the length of A, B, not counting spaces or special characters
        A_length = data_utils.count_length_no_special(A)
        B_length = data_utils.count_length_no_special(B)

        # Initialize embeddings with zeros
        emb_A = np.zeros(768)
        emb_B = np.zeros(768)
        emb_P = np.zeros(768)

        # Initialize counts
        count_chars = 0
        cnt_A, cnt_B, cnt_P = 0, 0, 0

        features = pd.DataFrame(bert_output.loc[i,"features"]) # Get the BERT embeddings for the current line in the data file
        for j in range(2,len(features)):  # Iterate over the BERT tokens for the current line; we skip over the first 2 tokens, which don't correspond to words
            token = features.loc[j,"token"]

            # See if the character count until the current token matches the offset of any of the 3 target words
            if count_chars  == P_offset: 
                # print(token)
                emb_P += np.array(features.loc[j,"layers"][0]['values'])
                cnt_P += 1
            if count_chars in range(A_offset, A_offset + A_length): 
                # print(token)
                emb_A += np.array(features.loc[j,"layers"][0]['values'])
                cnt_A +=1
            if count_chars in range(B_offset, B_offset + B_length): 
                # print(token)
                emb_B += np.array(features.loc[j,"layers"][0]['values'])
                cnt_B +=1                               
            # Update the character count
            count_chars += data_utils.count_length_no_special(token)
        # Taking the average between tokens in the span of A or B, so divide the current value by the count 
        emb_A /= cnt_A
        emb_B /= cnt_B

        # Work out the label of the current piece of text
        label = "Neither"
        if (df.loc[i,"A-coref"] == True):
            label = "A"
        if (df.loc[i,"B-coref"] == True):
            label = "B"

        # Put everything together in emb
        emb.iloc[i] = [emb_A, emb_B, emb_P, label]

    return emb


def parse_json(embeddings):
    '''
    Taken from public Kaggle kernal: Taming the BERT - a baseline

    Parses the embeddigns given by BERT, and suitably formats them to be passed to the MLP model

    Input: embeddings, a DataFrame containing contextual embeddings from BERT, as well as the labels for the classification problem
    columns: "emb_A": contextual embedding for the word A
             "emb_B": contextual embedding for the word B
             "emb_P": contextual embedding for the pronoun
             "label": the answer to the coreference problem: "A", "B" or "NEITHER"

    Output: X, a numpy array containing, for each line in the GAP file, the concatenation of the embeddings of the target words
            Y, a numpy array containing, for each line in the GAP file, the one-hot encoded answer to the coreference problem
    '''
    embeddings.sort_index(inplace = True) # Sorting the DataFrame, because reading from the json file messed with the order
    X = np.zeros((len(embeddings),3*768))
    Y = np.zeros((len(embeddings), 3))

    # Concatenate features
    for i in range(len(embeddings)):
        A = np.array(embeddings.loc[i,"emb_A"])
        B = np.array(embeddings.loc[i,"emb_B"])
        P = np.array(embeddings.loc[i,"emb_P"])
        X[i] = np.concatenate((A,B,P))

    # One-hot encoding for labels
    for i in range(len(embeddings)):
        label = embeddings.loc[i,"label"]
        if label == "A":
            Y[i,0] = 1
        elif label == "B":
            Y[i,1] = 1
        else:
            Y[i,2] = 1

    return X, Y
