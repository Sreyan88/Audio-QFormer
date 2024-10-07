import pandas as pd

file_path = "/fs/nexus-projects/brain_project/CLAP/test_clotho_ck.csv"
df =  pd.read_csv(file_path, sep=',')

grouped_df = df.groupby('path').agg(lambda x: list(x)).reset_index()

grouped_df.to_csv("/fs/nexus-projects/brain_project/CLAP/test_clotho_rt_ck.csv", sep=",", index=False)
