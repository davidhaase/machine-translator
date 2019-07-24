import os
import re
import pickle
from datetime import datetime as dt

import boto3
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
        finally:
            os.remove(file_name)

    def write(self, file_name):
        self.bucket.upload_file(file_name,Key=file_name)

    def list_dir(self, path):
        return [obj.key for obj in self.bucket.objects.filter(Prefix=path)]


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
