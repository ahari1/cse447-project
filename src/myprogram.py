#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import load_bloom, next_char as nc, init_pool
import time
import matplotlib.pyplot as plt

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, model=None, token_vocab=None, token_trie=None):
        self.model = model
        self.token_vocab = token_vocab
        self.token_trie = token_trie
        self.common_chars = [" ", "e", 'a', 'r', 'i', 'n', 's']
        

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line.strip('\r\n')
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # initialize multiprocessing pool
        pool = init_pool(self.token_vocab, self.token_trie)
        batch_size = 32

        # parallelize this at some point
        preds: list[str] = []
        eval_times = []
        filtering_times = []
        input_lengths = []

        for i in range(0, len(data), batch_size):
            batch = data[i:min(i + batch_size, len(data))]
            batch_results = nc(self.model, pool, self.token_vocab, self.token_trie, batch)
            try:
                curr_eval_times = []
                curr_filtering_times = []
                batch_preds = []
                for j, (curr_preds, eval_time, filtering_time) in enumerate(batch_results):
                    curr_preds = [p[0] for p in curr_preds if p[0] not in "\u000A\u000B\u000C\u000D\u0085\u2028\u2029"]
                    curr_eval_times.append(eval_time)
                    curr_filtering_times.append(filtering_time)

                    if len(curr_preds) < 3:
                        for i in range(len(self.common_chars)):
                            if self.common_chars[i] not in preds:
                                curr_preds.append(self.common_chars[i])
                            if len(curr_preds) >=3:
                                break
                    else:
                        curr_preds = curr_preds[:3]

                    batch_preds.append(curr_preds)
            except:
                batch_preds = [[" ", "e", "a"]] * len(batch)

            preds.extend(''.join(pred) for pred in batch_preds)
            eval_times.extend(curr_eval_times)
            filtering_times.extend(curr_filtering_times)
            input_lengths.extend(len(batch[j]) for j in range(i, min(i+batch_size, len(data))))

        # close the pool
        pool.close()
        pool.join()
        # Plot the time taken for each prediction with respect to the number of characters in the input

        return preds, eval_times, filtering_times

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        
        model, token_vocab, token_trie = load_bloom(work_dir)
        instance = cls(model, token_vocab, token_trie)
        # instance.model = model
        # instance.token_vocab = token_vocab
        # with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        #     dummy_save = f.read()
        return instance


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel(None, None)
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
