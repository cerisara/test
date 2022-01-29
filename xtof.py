DATADIR = "/home/xtof/nas1/TALC/Synalp/Corpus/MMUL/data/test/"
files = ("world_religions_test.csv",)

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-t5-large")

nok,nrep,ntot=0,0,0
for i in range(test_df.shape[0]):
    prompt_end = format_example(test_df, i, include_answer=False)
    label = test_df.iloc[i, test_df.shape[1]-1]
    print()
    gold = ord(label)-ord('A')
    print(label,gold)
    print(prompt_end)
    s = tokenizer.encode(prompt_end,return_tensors='pt')
    y = model.generate(s)
    sy = tokenizer.decode(y[0])
    sy = sy.replace('<pad>','').replace("</s>","").strip()
    opts = prompt_end.split('\n')
    rep=-1
    for j,s in enumerate(opts):
        if sy in s: rep=ord(s[1])-ord('a')
    ntot+=1
    if rep>=0:
        nrep+=1
        if gold==rep: nok+=1
    acc=float(nok)/float(ntot)
    ans=float(nrep)/float(ntot)
    print(sy,"ACC=",acc,ans,rep)

