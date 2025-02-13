import re
import matplotlib.pyplot as plt

data_text = """
2 characters: Success rate 50 / 185 = 0.2702702702702703.
3 characters: Success rate 56 / 178 = 0.3146067415730337.
4 characters: Success rate 66 / 177 = 0.3728813559322034.
5 characters: Success rate 88 / 170 = 0.5176470588235295.
6 characters: Success rate 86 / 164 = 0.524390243902439.
7 characters: Success rate 110 / 156 = 0.7051282051282052.
8 characters: Success rate 105 / 150 = 0.7.
9 characters: Success rate 99 / 141 = 0.7021276595744681.
10 characters: Success rate 107 / 135 = 0.7925925925925926.
11 characters: Success rate 104 / 129 = 0.8062015503875969.
12 characters: Success rate 104 / 125 = 0.832.
13 characters: Success rate 96 / 120 = 0.8.
14 characters: Success rate 95 / 115 = 0.8260869565217391.
15 characters: Success rate 89 / 109 = 0.8165137614678899.
16 characters: Success rate 84 / 99 = 0.8484848484848485.
17 characters: Success rate 72 / 97 = 0.7422680412371134.
18 characters: Success rate 67 / 91 = 0.7362637362637363.
19 characters: Success rate 79 / 89 = 0.8876404494382022.
20 characters: Success rate 72 / 86 = 0.8372093023255814.
21 characters: Success rate 65 / 84 = 0.7738095238095238.
22 characters: Success rate 63 / 84 = 0.75.
23 characters: Success rate 65 / 80 = 0.8125.
24 characters: Success rate 65 / 75 = 0.8666666666666667.
25 characters: Success rate 60 / 74 = 0.8108108108108109.
26 characters: Success rate 58 / 70 = 0.8285714285714286.
27 characters: Success rate 50 / 63 = 0.7936507936507936.
28 characters: Success rate 55 / 63 = 0.873015873015873.
29 characters: Success rate 53 / 62 = 0.8548387096774194.
30 characters: Success rate 53 / 58 = 0.9137931034482759.
31 characters: Success rate 45 / 56 = 0.8035714285714286.
32 characters: Success rate 46 / 54 = 0.8518518518518519.
33 characters: Success rate 44 / 51 = 0.8627450980392157.
34 characters: Success rate 43 / 49 = 0.8775510204081632.
35 characters: Success rate 42 / 47 = 0.8936170212765957.
36 characters: Success rate 40 / 45 = 0.8888888888888888.
37 characters: Success rate 37 / 44 = 0.8409090909090909.
38 characters: Success rate 37 / 43 = 0.8604651162790697.
39 characters: Success rate 31 / 38 = 0.8157894736842105.
40 characters: Success rate 34 / 38 = 0.8947368421052632.
41 characters: Success rate 30 / 37 = 0.8108108108108109.
42 characters: Success rate 25 / 36 = 0.6944444444444444.
43 characters: Success rate 31 / 35 = 0.8857142857142857.
44 characters: Success rate 25 / 30 = 0.8333333333333334.
45 characters: Success rate 24 / 28 = 0.8571428571428571.
46 characters: Success rate 24 / 28 = 0.8571428571428571.
47 characters: Success rate 23 / 27 = 0.8518518518518519.
48 characters: Success rate 21 / 26 = 0.8076923076923077.
49 characters: Success rate 21 / 25 = 0.84.
50 characters: Success rate 23 / 24 = 0.9583333333333334.
51 characters: Success rate 18 / 23 = 0.782608695652174.
52 characters: Success rate 17 / 21 = 0.8095238095238095.
53 characters: Success rate 14 / 19 = 0.7368421052631579.
54 characters: Success rate 14 / 19 = 0.7368421052631579.
55 characters: Success rate 16 / 19 = 0.8421052631578947.
56 characters: Success rate 15 / 17 = 0.8823529411764706.
57 characters: Success rate 14 / 17 = 0.8235294117647058.
58 characters: Success rate 17 / 17 = 1.0.
59 characters: Success rate 14 / 17 = 0.8235294117647058.
60 characters: Success rate 13 / 15 = 0.8666666666666667.
61 characters: Success rate 12 / 14 = 0.8571428571428571.
62 characters: Success rate 11 / 12 = 0.9166666666666666.
63 characters: Success rate 9 / 11 = 0.8181818181818182.
64 characters: Success rate 9 / 11 = 0.8181818181818182.
65 characters: Success rate 11 / 11 = 1.0.
66 characters: Success rate 8 / 9 = 0.8888888888888888.
67 characters: Success rate 7 / 8 = 0.875.
68 characters: Success rate 8 / 8 = 1.0.
69 characters: Success rate 7 / 8 = 0.875.
70 characters: Success rate 7 / 8 = 0.875.
71 characters: Success rate 7 / 8 = 0.875.
72 characters: Success rate 7 / 8 = 0.875.
73 characters: Success rate 8 / 8 = 1.0.
74 characters: Success rate 8 / 8 = 1.0.
75 characters: Success rate 6 / 8 = 0.75.
76 characters: Success rate 6 / 7 = 0.8571428571428571.
77 characters: Success rate 6 / 7 = 0.8571428571428571.
78 characters: Success rate 6 / 6 = 1.0.
79 characters: Success rate 5 / 6 = 0.8333333333333334.
80 characters: Success rate 5 / 6 = 0.8333333333333334.
81 characters: Success rate 3 / 6 = 0.5.
82 characters: Success rate 5 / 5 = 1.0.
83 characters: Success rate 4 / 5 = 0.8.
84 characters: Success rate 4 / 5 = 0.8.
85 characters: Success rate 5 / 5 = 1.0.
86 characters: Success rate 4 / 5 = 0.8.
"""

# Extract number of characters (x) and success rate (y) from each line
x_vals, y_vals = [], []
lines = data_text.strip().split('\n')

for line in lines:
    match = re.search(r'^(\d+)\s+characters:\s+Success rate\s+(\d+)\s*/\s*(\d+)\s*=\s*([\d\.]+)', line)
    if match:
        print(match.group(4))
        x_vals.append(int(match.group(1)))
        y_vals.append(float(match.group(4)[:-1]))

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals)
plt.title("Number of Characters vs. Success Rate")
plt.xlabel("Number of Characters")
plt.ylabel("Success Rate")
plt.grid(True)
plt.show()
