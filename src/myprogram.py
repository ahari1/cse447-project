#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import load_bloom, next_char as nc
import time


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, model=None, token_vocab=None):
        self.model = model
        self.token_vocab = token_vocab
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
        t0 = time.time()
        # parallelize this at some point
        preds: list[str] = []
        k = 0
        t = time.time()
        for test in data:
            try:
                if k > 0 and k % 50 == 0:
                    print(f"{k} in {time.time() - t}s (average {(time.time() - t) / k}s per prediction).")
                k += 1
                curr_preds = nc(self.model, self.token_vocab, test)
                curr_preds = [p[0] for p in curr_preds if p[0] not in "\u000A\u000B\u000C\u000D\u0085\u2028\u2029"]

                if len(curr_preds) < 3:
                    for i in range(len(self.common_chars)):
                        if self.common_chars[i] not in preds:
                            curr_preds.append(self.common_chars[i])
                        if len(curr_preds) >=3:
                            break
                else:
                    curr_preds = curr_preds[:3]
            except:
                curr_preds = [" ", "e", "a"]

            preds.append(''.join(curr_preds))
        print(f"On average took {(time.time() - t0) / len(preds) * 1000} ms per prediction.")
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        
        model, token_vocab = load_bloom(work_dir)
        instance = cls(model, token_vocab)
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
