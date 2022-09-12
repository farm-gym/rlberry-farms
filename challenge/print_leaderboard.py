from tabulate import tabulate
import pandas as pd

df = pd.read_csv("/home/challenge_env/rlberry-farms/challenge/leaderboard.csv", index_col=0)
print(tabulate(df, headers='keys', tablefmt='psql'))
