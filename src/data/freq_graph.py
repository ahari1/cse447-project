import re
import matplotlib.pyplot as plt

data_text = """
2 characters: Success rate 77 / 185 = 0.41621621621621624.
3 characters: Success rate 112 / 178 = 0.6292134831460674.
4 characters: Success rate 116 / 177 = 0.655367231638418.
5 characters: Success rate 132 / 170 = 0.7764705882352941.
6 characters: Success rate 126 / 164 = 0.7682926829268293.
7 characters: Success rate 134 / 156 = 0.8589743589743589.
8 characters: Success rate 122 / 150 = 0.8133333333333334.
9 characters: Success rate 110 / 141 = 0.7801418439716312.
10 characters: Success rate 109 / 135 = 0.8074074074074075.
11 characters: Success rate 109 / 129 = 0.8449612403100775.
12 characters: Success rate 103 / 125 = 0.824.
13 characters: Success rate 104 / 120 = 0.8666666666666667.
14 characters: Success rate 99 / 115 = 0.8608695652173913.
15 characters: Success rate 87 / 109 = 0.7981651376146789.
16 characters: Success rate 88 / 99 = 0.8888888888888888.
17 characters: Success rate 76 / 97 = 0.7835051546391752.
18 characters: Success rate 74 / 91 = 0.8131868131868132.
19 characters: Success rate 79 / 89 = 0.8876404494382022.
20 characters: Success rate 71 / 86 = 0.8255813953488372.
21 characters: Success rate 68 / 84 = 0.8095238095238095.
22 characters: Success rate 67 / 84 = 0.7976190476190477.
23 characters: Success rate 64 / 80 = 0.8.
24 characters: Success rate 64 / 75 = 0.8533333333333334.
25 characters: Success rate 61 / 74 = 0.8243243243243243.
26 characters: Success rate 60 / 70 = 0.8571428571428571.
27 characters: Success rate 56 / 63 = 0.8888888888888888.
28 characters: Success rate 58 / 63 = 0.9206349206349206.
29 characters: Success rate 53 / 62 = 0.8548387096774194.
30 characters: Success rate 52 / 58 = 0.896551724137931.
31 characters: Success rate 46 / 56 = 0.8214285714285714.
32 characters: Success rate 49 / 54 = 0.9074074074074074.
33 characters: Success rate 43 / 51 = 0.8431372549019608.
34 characters: Success rate 45 / 49 = 0.9183673469387755.
35 characters: Success rate 42 / 47 = 0.8936170212765957.
36 characters: Success rate 40 / 45 = 0.8888888888888888.
37 characters: Success rate 39 / 44 = 0.8863636363636364.
38 characters: Success rate 39 / 43 = 0.9069767441860465.
39 characters: Success rate 34 / 38 = 0.8947368421052632.
40 characters: Success rate 37 / 38 = 0.9736842105263158.
41 characters: Success rate 32 / 37 = 0.8648648648648649.
42 characters: Success rate 27 / 36 = 0.75.
43 characters: Success rate 33 / 35 = 0.9428571428571428.
44 characters: Success rate 26 / 30 = 0.8666666666666667.
45 characters: Success rate 25 / 28 = 0.8928571428571429.
46 characters: Success rate 23 / 28 = 0.8214285714285714.
47 characters: Success rate 26 / 27 = 0.9629629629629629.
48 characters: Success rate 21 / 26 = 0.8076923076923077.
49 characters: Success rate 21 / 25 = 0.84.
50 characters: Success rate 23 / 24 = 0.9583333333333334.
51 characters: Success rate 19 / 23 = 0.8260869565217391.
52 characters: Success rate 18 / 21 = 0.8571428571428571.
53 characters: Success rate 17 / 19 = 0.8947368421052632.
54 characters: Success rate 14 / 19 = 0.7368421052631579.
55 characters: Success rate 16 / 19 = 0.8421052631578947.
56 characters: Success rate 16 / 17 = 0.9411764705882353.
57 characters: Success rate 14 / 17 = 0.8235294117647058.
58 characters: Success rate 17 / 17 = 1.0.
59 characters: Success rate 16 / 17 = 0.9411764705882353.
60 characters: Success rate 14 / 15 = 0.9333333333333333.
61 characters: Success rate 13 / 14 = 0.9285714285714286.
62 characters: Success rate 11 / 12 = 0.9166666666666666.
63 characters: Success rate 10 / 11 = 0.9090909090909091.
64 characters: Success rate 9 / 11 = 0.8181818181818182.
65 characters: Success rate 11 / 11 = 1.0.
66 characters: Success rate 9 / 9 = 1.0.
67 characters: Success rate 7 / 8 = 0.875.
68 characters: Success rate 8 / 8 = 1.0.
69 characters: Success rate 7 / 8 = 0.875.
70 characters: Success rate 7 / 8 = 0.875.
71 characters: Success rate 7 / 8 = 0.875.
72 characters: Success rate 7 / 8 = 0.875.
73 characters: Success rate 8 / 8 = 1.0.
74 characters: Success rate 8 / 8 = 1.0.
75 characters: Success rate 7 / 8 = 0.875.
76 characters: Success rate 7 / 7 = 1.0.
77 characters: Success rate 6 / 7 = 0.8571428571428571.
78 characters: Success rate 6 / 6 = 1.0.
79 characters: Success rate 4 / 6 = 0.6666666666666666.
80 characters: Success rate 6 / 6 = 1.0.
81 characters: Success rate 4 / 6 = 0.6666666666666666.
82 characters: Success rate 5 / 5 = 1.0.
83 characters: Success rate 5 / 5 = 1.0.
84 characters: Success rate 5 / 5 = 1.0.
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

fig = plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals)
plt.title("Number of Characters vs. Success Rate")
plt.xlabel("Number of Characters")
plt.ylabel("Success Rate")
plt.grid(True)
fig.savefig("llama_results.png")
