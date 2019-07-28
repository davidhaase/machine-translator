import os
import pickle

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

    def write(self, file_name):
        self.bucket.upload_file(file_name,Key=file_name)

    def list_dir(self, path):
        return [obj.key for obj in self.bucket.objects.filter(Prefix=path)]
