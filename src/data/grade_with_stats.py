# How to use:
# Just run something like this:
# `python src/data/grade_with_stats.py data/en/pred.txt data/en/answer.txt data/en/input.txt`

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('fpred')
parser.add_argument('fgold')
parser.add_argument('forig')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()


def load_pred(fname, force_limit=None):
    with open(fname) as f:
        loaded = []
        for line in f:
            line = line[:-1].lower()
            if force_limit is not None:
                line = line[:force_limit]
            loaded.append(line)
        return loaded


orig = load_pred(args.forig)
pred = load_pred(args.fpred, force_limit=3)
gold = load_pred(args.fgold)

if len(pred) < len(gold):
    pred.extend([''] * (len(gold) - len(pred)))

correct = 0
len_to_correct = dict()
for i, (p, g) in enumerate(zip(pred, gold)):
    right = g in p
    correct += right
    k = len(orig[i])
    if k not in len_to_correct.keys():
        len_to_correct[k] = [right, 1]
    else:
        len_to_correct[k][0] += right
        len_to_correct[k][1] += 1
    if args.verbose:
        print('Input {}: {}, {} is {} in {}'.format(i, 'right' if right else 'wrong', g, 'in' if right else 'not in', p))
print('Success rate: {}'.format(correct/len(gold)))

for k, v in len_to_correct.items():
    if v[1] >= 5:
        print(f"{k} characters: Success rate {v[0]} / {v[1]} = {v[0] / v[1]}.")
