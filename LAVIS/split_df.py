import pandas as pd

df = pd.read_csv('/fs/nexus-projects/brain_project/CLAP/val.csv', sep=',', header=0)

filtered_df = df[df['dataset'] == "audiocaps"]

filtered_df['end'] = pd.NA

filtered_df['split_name'] = 'val'

filtered_df.to_csv("/fs/nexus-projects/brain_project/naacl_audio/audiocaps_from_val.csv", sep=",", index=False)