DATADIR = "/home/xtof/nas1/TALC/Synalp/Corpus/MMUL/data/test/"
files = ("world_religions_test.csv",)

import pandas as pd

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n({}) {}".format(choices[j], df.iloc[idx, j+1])
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt.lower()

choices = ["A", "B", "C", "D"]
test_df = pd.read_csv(DATADIR+files[0], header=None)
answers = choices[:test_df.shape[1]-2]
print(test_df.shape)
print(answers)
for i in range(test_df.shape[0]):
    prompt_end = format_example(test_df, i, include_answer=False)
    label = test_df.iloc[i, test_df.shape[1]-1]
    print()
    print(label)
    print(prompt_end)

