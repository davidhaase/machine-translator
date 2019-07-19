import os
import re

class Logger():
    def __init__(self, log_dir, log_prefix='log_', ext='log'):
        self.ext = ext
        self.log_prefix = log_prefix

        if (os.path.isdir(log_dir) is False):

            # if there is no log directory
            # ...create one and start at log_001.log
            try:
                os.makedirs(log_dir)
                self.log_dir = log_dir
            except Exception as e:
                print(e)
                return

        pattern = re.compile(r'log_(\d+)')
        
        max_dir = 0
        for dir_name in os.listdir(write_dir):
            match = pattern.search(dir_name)
            if match:
                count = int(match.group(1))
                if count > max_dir:
                    max_dir = count
        file_path = '{}{}001.{}'.format(log_dir, log_prefix, ext)
        new_dir = 'feat_{0:03d}'.format(max_dir + 1)
        return new_dir


    # def get_next_dir(self, write_dir):
    #     pattern = re.compile(r'feat_(\d+)')
    #
    #     max_dir = 0
    #     for dir_name in os.listdir(write_dir):
    #         match = pattern.search(dir_name)
    #         if match:
    #             count = int(match.group(1))
    #             if count > max_dir:
    #                 max_dir = count
    #
    #     new_dir = 'feat_{0:03d}'.format(max_dir + 1)
    #     return new_dir
