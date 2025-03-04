# How to use:
# Go to https://opus.nlpl.eu/OpenSubtitles/fi&en/v2018/OpenSubtitles
# Download some dataset into data/en.txt
# `python src/data/parse_opus_data.py data/en.txt`
# `python src/myprogram.py test --work_dir work --test_data data/en/input.txt --test_output data/en/pred.txt`
# `python grader/grade.py data/en/pred.txt data/en/answer.txt`

import os
from argparse import ArgumentParser
import random

parser = ArgumentParser()
parser.add_argument('name')
args = parser.parse_args()

if __name__ == '__main__':
    assert isinstance(args.name, str)
    dirname = args.name[:len(args.name) - 4]

    with open(args.name, "r") as f:
        lines = f.readlines()

    xs = []
    ys = []

    for i in range(min(20000, len(lines))):
        line = lines[i].strip()
        check = True
        while check:
            check = False
            if line.startswith(' - '):
                check = True
                line = line[3:]
            if line.startswith('- '):
                check = True
                line = line[2:]
            if line.startswith('('):
                check = True
                line = line[line.index(')') + 1:]
        line = line.replace("n' t ", "n't ")
        line = line.replace("n' t,", "n't,")
        line = line.replace("n' t.", "n't.")
        line = line.replace("' re ", "'re ")
        line = line.replace("' s ", "'s ")
        line = line.replace("' s,", "'s,")
        line = line.replace("' s.", "'s.")
        line = line.replace("I' m ", "I'm ")
        line = line.replace("' il ", "'ll ")
        line = line.replace('- ', '-')
        line = line.strip()
        if len(line) < 15:
            continue
        j = int(random.random() * (len(line) - 3)) + 3
        xs.append(line[:j])
        ys.append(line[j])

    xstr = '\n'.join(xs)
    ystr = '\n'.join(ys)

    os.makedirs(dirname, exist_ok=True)

    xname = os.path.join(dirname, "input.txt")
    with open(xname, "w") as f:
        f.write(xstr)

    yname = os.path.join(dirname, "answer.txt")
    with open(yname, "w") as f:
        f.write(ystr)
