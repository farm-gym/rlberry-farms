from tabulate import tabulate
import pandas as pd
import sys
import os

CHALLENGE_DIR = "/challenge_bin"
DATA_LOC = os.environ.get('challenge_data_dir')

if len(sys.argv)>1:
    toprint = sys.argv[1]
else:
    toprint = 'default'

if toprint == "default":
    df = pd.read_csv(
        os.path.join(DATA_LOC, "leaderboard.csv"), index_col=0
    )
elif toprint == '-a':
    df = pd.read_csv(
        os.path.join(DATA_LOC, "all_scores.csv"), index_col=0
    )
elif toprint == '-m':
    df = pd.read_csv(
        os.path.join(DATA_LOC, "all_scores.csv"), index_col=0
    )
    df = df.loc[df["name"]==os.getlogin()]

print(tabulate(df, headers="keys", tablefmt="psql"))
