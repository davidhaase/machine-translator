import os
import string
import re
import pickle
from datetime import datetime as dt
from unicodedata import normalize

import boto3

from numpy import array
from numpy.random import rand
from numpy.random import shuffle
from numpy import argmax

from keras.preprocessing.text import Tokenizer

from credentials import aws_access_key_id, aws_secret_access_key

class S3Bucket():
    def __init__(self, bucket_name='flatiron-audio-classification'):
        self.bucket_name = bucket_name

        try:
            self.resource = boto3.resource('s3',aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            self.client = boto3.client('s3',aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            self.bucket = self.resource.Bucket(bucket_name)
        except Exception as e:
            print(e)


    def read(self, file_name, encoding='utf-8'):
        obj = self.resource.Object(self.bucket_name, file_name)
        if (encoding is None):
            return obj.get()['Body'].read()
        else:
            return obj.get()['Body'].read().decode(encoding)


    def read_lines(self, file_name, encoding='utf-8'):
        obj = self.client.get_object(Bucket=self.bucket_name, Key=file_name)
        return [line.decode(encoding) for line in obj['Body'].read().splitlines()]

    def load(self, file_name):
        pickle_path = 'Pickles/'
        try:
            f = open(file_name, 'wb')
            self.client.download_fileobj(self.bucket_name, pickle_path + file_name, f)
            f.close()
            pkl = pickle.load(open(file_name, 'rb'))
            os.remove(file_name)
            return pkl
        except Exception as e:
            print('failed here', e)



    def dump(self, data, file_name):
        try:
            pickle.dump(data, open(file_name, 'wb'))
            pickle_path = 'Pickles/'
            self.bucket.upload_file(file_name,Key=pickle_path + file_name)

        except Exception as e:
            print(e)

    def write(self, file_name):
        self.bucket.upload_file(file_name,Key=file_name)

    def list_dir(self, path):
        return [obj.key for obj in self.bucket.objects.filter(Prefix=path)]

class Sentences():
    def __init__(self, L1, L2):
        self.prefix = L1 + '2' + L2
        self.L1 = L1
        self.L2 = L2
        self.s3 = S3Bucket()
        self.source = ''
        self.sent_count = 0
        self.clean_pairs = []
        self.clean_pair_file = 'pickles/' + self.prefix + '_sentence_pairs.pkl'
        self.train_file = 'pickles/' + self.prefix + '_train.pkl'
        self.test_file = 'pickles/' + self.prefix + '_test.pkl'
        self.dataset = None
        self.train = None
        self.train_X = None
        self.train_y = None
        self.test = None
        self.test_X = None
        self.test_y = None
        self.L1_tokenizer = None
        self.L1_length = None
        self.L1_vocab_size = None
        self.L2_tokenizer = None
        self.L2_length = None
        self.L2_vocab_size = None

    def spot_check(self, n=100):
        if len(self.clean_pairs) > 0:
            n = n if len(self.clean_pairs) >= n else len(self.clean_pairs)
            for i in range(n):
                print('{} => {}'.format(self.clean_pairs[i,0], self.clean_pairs[i,1]))
        else:
            print('No data (sentence pairs) found'.format(len(self.clean_pairs)))

    def load_pairs(self, source='', force_rebuild=False):

        # Run through the following series of checks to see if data already exists
        if force_rebuild==False:
            if len(self.clean_pairs) > 0:
                # There is already data loaded, so use that
                print('{} pairs already in memory'.format(len(self.clean_pairs)))
                return

            if os.path.isfile(self.clean_pair_file):
                # There is still a pickle file locally, so reload that into memory
                self.clean_pairs = pickle.load(open(self.clean_pair_file, 'rb'))
                print('Loading pairs from existing file: {}'.format(self.clean_pair_file))
                return

            if source == self.source:
                print('Specified source already exists, force_rebuild=True to force a rebuild')
                return

        if source == '':
            # And finally, there's no source give to rebuild
            print('Cannot rebuild pairs. No source detected or provided')
            return

        self.source = source
        text = self.s3.read(source)
        # split into english-german pairs
        lines = text.strip().split('\n')
        pairs = [line.split('\t') for line in  lines]
        clean_pairs = list()
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))

        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        cleaned = list()
        for pair in pairs:
            clean_pair = list()
            for line in pair:
                # normalize unicode characters
                line = normalize('NFD', line).encode('ascii', 'ignore')
                line = line.decode('UTF-8')
                # tokenize on white space
                line = line.split()
                # convert to lowercase
                line = [word.lower() for word in line]
                # remove punctuation from each token
                line = [word.translate(table) for word in line]
                # remove non-printable chars form each token
                line = [re_print.sub('', w) for w in line]
                # remove tokens with numbers in them
                line = [word for word in line if word.isalpha()]
                # store as string
                clean_pair.append(' '.join(line))
            cleaned.append(clean_pair)
        self.clean_pairs = array(cleaned)
        self.s3.dump(self.clean_pairs, self.clean_pair_file)
        print('Saved {} pairs: {}'.format(len(self.clean_pairs), self.clean_pair_file))

        return self.clean_pairs

    def create_datasets(self, split=90, sent_count=10000):
        pair_count = len(self.clean_pairs)
        self.sent_count = sent_count
        if pair_count < sent_count:
            print('Cannot create datasets. {} pairs requested, but only {} found'.format(sent_count, pair_count))
            return

        dataset = self.clean_pairs[:sent_count, :]
        # random shuffle
        shuffle(dataset)
        # split into train/test
        self.train, self.test, self.dataset = dataset[:9000], dataset[9000:], dataset

        def create_tokenizer(lines):
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(lines)
            return tokenizer

        def max_length(lines):
            return max(len(line.split()) for line in lines)

        # prepare english tokenizer
        self.L1_tokenizer = create_tokenizer(dataset[:, 1])
        self.L1_length = max_length(dataset[:, 1])
        self.L1_vocab_size = len(self.L1_tokenizer.word_index) + 1
        self.L2_tokenizer = create_tokenizer(dataset[:, 0])
        self.L2_length = max_length(dataset[:, 0])
        self.L2_vocab_size = len(self.L2_tokenizer.word_index) + 1
        print('English Vocabulary Size: {}'.format(str(self.L2_vocab_size)))
        print('English Max Length: {}'.format(str(self.L2_length)))
        print('German Vocabulary Size: {}'.format(str(self.L1_vocab_size)))
        print('German Max Length: {}'.format(str(self.L1_length)))



class Logger():
    def __init__(self, log_location, log_prefix='log_', ext='log'):
        self.ext = ext
        self.log_prefix = log_prefix

        is_dir_name = True if log_location[-1] is '/' else False

        # Check to see if the desired log location exists as a directory
        if is_dir_name:

            # Make the log directory if it doesn't exist already
            if os.path.isdir(log_location) is False:
                try:
                    os.makedirs(log_location)
                except Exception as e:
                    print(e)
                    return

            # Now determine the next file number
            # ... and create the next log file
            self.log_path = log_location
            pattern = re.compile(log_prefix + r'(\d+)\.' + ext)

            max_file = 0
            for f in os.listdir(log_location):
                match = pattern.search(f)
                if match:
                    count = int(match.group(1))
                    if count > max_file:
                        max_file = count
            next_file = '{0:03d}'.format(max_file + 1)
            file_name  = '{}{}{}.{}'.format(log_location, log_prefix, next_file, ext)
            try:
                f = open(file_name, 'w+')
                f.write('')
                f.close()
                self.file_name = file_name

            except Exception as e:
                print(e)
                return

        # So, they passed a direct filename for the log
        else:
            try:
                f = open(log_location, 'w+')
                f.write('')
                f.close()
                self.file_name = log_location
                self.log_path = '/'.join(log_location.split('/')[:-1])

            except Exception as e:
                print(e)
                return

        self.write('Log started')

    def write(self, text):
        prefix = '[{}] '.format(dt.now())
        try:
            f = open(self.file_name, 'w+')
            text_line = prefix + text
            f.write(text_line)

        except Exception as e:
            print(e)
        finally:
            f.close()
