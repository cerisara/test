DATADIR = "/home/xtof/nas1/TALC/Synalp/Corpus/MMUL/data/test/"
files = ("world_religions_test.csv",
"abstract_algebra_test.csv",
"anatomy_test.csv",
"astronomy_test.csv",
"business_ethics_test.csv",
"clinical_knowledge_test.csv",
"college_biology_test.csv",
"college_chemistry_test.csv",
"college_computer_science_test.csv",
"college_mathematics_test.csv",
"college_medicine_test.csv",
"college_physics_test.csv",
"computer_security_test.csv",
"conceptual_physics_test.csv",
"econometrics_test.csv",
"electrical_engineering_test.csv",
"elementary_mathematics_test.csv",
"formal_logic_test.csv",
"global_facts_test.csv",
"high_school_biology_test.csv",
"high_school_chemistry_test.csv",
"high_school_computer_science_test.csv",
"high_school_european_history_test.csv",
"high_school_geography_test.csv",
"high_school_government_and_politics_test.csv",
"high_school_macroeconomics_test.csv",
"high_school_mathematics_test.csv",
"high_school_microeconomics_test.csv",
"high_school_physics_test.csv",
"high_school_psychology_test.csv",
"high_school_statistics_test.csv",
"high_school_us_history_test.csv",
"high_school_world_history_test.csv",
"human_aging_test.csv",
"human_sexuality_test.csv",
"international_law_test.csv",
"jurisprudence_test.csv",
"logical_fallacies_test.csv",
"machine_learning_test.csv",
"management_test.csv",
"marketing_test.csv",
"medical_genetics_test.csv",
"miscellaneous_test.csv",
"moral_disputes_test.csv",
"moral_scenarios_test.csv",
"nutrition_test.csv",
"philosophy_test.csv",
"prehistory_test.csv",
"professional_accounting_test.csv",
"professional_law_test.csv",
"professional_medicine_test.csv",
"professional_psychology_test.csv",
"public_relations_test.csv",
"security_studies_test.csv",
"sociology_test.csv",
"us_foreign_policy_test.csv",
"virology_test.csv",
"world_religions_test.csv")

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
tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-t5-large")
nok,nrep,ntot=0,0,0

for fil in files:
    print("file",fil)
    test_df = pd.read_csv(DATADIR+fil, header=None)
    answers = choices[:test_df.shape[1]-2]
    print(test_df.shape)
    print(answers)

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


