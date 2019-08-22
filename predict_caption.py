print("loading caption generating modules")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import json
import numpy as np
import tensorflow as tf
print("loaded caption generating modules")

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""

    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text

    def captions_to_tokens(self, captions_list):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """

        # Note that text_to_sequences() takes a list of texts.
        tokens = self.texts_to_sequences(captions_list)

        return tokens


class PredictCaption():

    def __init__(self):
        print("loading model")
        self.session=tf.Session()
        self.graph=tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():
                self.model=load_model("weights-glove-15.hdf5")
        print("loaded model")


        with open('training_label.json',encoding="utf8") as f:
            print("got file")
            data=json.load(f)
            print("got data")

        #convert list into numpy array
        data=np.asarray(data)

        start_seq="ssss "
        end_seq=" eeee"
        print("marking captions and flattening captions")
        marked_captions_train=self.mark_captions(data,start_seq,end_seq)
        flatten_captions_train=self.flatten(marked_captions_train)
        #maximum number of words in the tokenizer
        vocab_size=10000
        #maximum length of a caption
        max_length=15
        print("creating tokenizer")
        self.tokenizer = TokenizerWrap(texts=flatten_captions_train,
                                  num_words=vocab_size)

        #check the token of start sequence
        token_start = self.tokenizer.word_index[start_seq.strip()]
        print(token_start)
        #check the token of end sequence
        token_end = self.tokenizer.word_index[end_seq.strip()]
        print(token_end)


    #create function to add start and end sequences
    def mark_captions(self,captions_listlist,start_seq,end_seq):
        marked_cap=[]
        for captions_list in captions_listlist:
            captions={'id':captions_list['id']}
            cap_list=[]
            for cap in captions_list['caption']:
                cap_list.append(start_seq+cap+end_seq)
            captions['caption']=cap_list
            marked_cap.append(captions)

        return marked_cap

    #create a function to flatten the list of lists
    def flatten(self,captions_listlist):
        captions_list = [caption
                         for captions_list in captions_listlist
                         for caption in captions_list['caption']]
        return captions_list


    def greedysearch(self,feat):
        start_seq="ssss "
        end_seq=" eeee"
        max_length=15
        in_text=start_seq
        feat_batch = np.expand_dims(feat, axis=0)

        for i in range(max_length):
            sequence= [self.tokenizer.word_index[w] for w in in_text.split()]
            sequence=pad_sequences([sequence], maxlen=max_length)
            with self.graph.as_default():
                with self.session.as_default():
                    yhat=self.model.predict([feat_batch, sequence], verbose=0)
            yhat=np.argmax(yhat)
            word=self.tokenizer.token_to_word(yhat)
            in_text+=' '+word

            if word == end_seq.strip():
                break

        final=in_text.split()
        final=final[1:-1]
        final=' '.join(final)
        return final
