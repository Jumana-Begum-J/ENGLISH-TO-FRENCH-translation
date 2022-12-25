from keras.models import load_model
import numpy as np
import pickle as pickle
from nltk.translate.bleu_score import sentence_bleu

encoder_path='models/encoder_modelPredTranslation.h5'
decoder_path='models/decoder_modelPredTranslation.h5'
charencoding_path='models/char2encoding.pkl'

def getChar2encoding(charencoding_path):
    f = open(charencoding_path, "rb")
    input_token_index = pickle.load(f)
    max_encoder_seq_length = pickle.load(f)
    num_encoder_tokens = pickle.load(f)
    reverse_target_char_index = pickle.load(f)
    num_decoder_tokens = pickle.load(f)
    target_token_index = pickle.load(f)
    f.close()
    return input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index

def encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

def decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index):
    # We encode the input
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    # We predict the output letter by letter 
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # We translate the token in hamain language
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # We check if it is the end of the string
        if (sampled_char == '\n' or
           len(decoded_sentence) > 500):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence

#Prediction
def start_prediction(sentence):
    #sentence="Hello"

    input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index= getChar2encoding(charencoding_path)
    encoder_input_data=encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens)
    encoder_model= load_model(encoder_path)
    decoder_model= load_model(decoder_path)

    input_seq = encoder_input_data

    decoded_sentence=decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index)
    print('-')
    print('Input sentence:', sentence)
    print('Decoded sentence:', decoded_sentence)

    #decoded_sentence = "I work"
    actual_output_sentence = "I work"
    predicted_output=['I', 'work']
    actual_output=actual_output_sentence.split()
    print(predicted_output)
    print(actual_output)
    bleuscore2= sentence_bleu(actual_output, predicted_output, weights=(1, 0 , 0, 0))
    print(bleuscore2)
    return decoded_sentence