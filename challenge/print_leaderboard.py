from tabulate import tabulate
import pandas as pd
import sys
import os

if len(sys.argv)>1:
    toprint = sys.argv[1]
else:
    toprint = 'default'

if toprint == "default":
    df = pd.read_csv(
        "/mnt/data/challenge/leaderboard.csv", index_col=0
    )
elif toprint == '-a':
    df = pd.read_csv(
        "/mnt/data/challenge/all_scores.csv", index_col=0
    )
elif toprint == '-m':
    df = pd.read_csv(
        "/mnt/data/challenge/all_scores.csv", index_col=0
    )
    df = df.loc[df["name"]==os.getlogin()]

print(tabulate(df, headers="keys", tablefmt="psql"))
