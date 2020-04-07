import pandas as pd
import os

here = os.path.dirname(os.path.abspath(__file__))
result_81 = pd.read_csv( os.path.join(here,"result_81.csv"))
result_80382 = pd.read_csv( os.path.join(here,"result_80382.csv"))
test_survived = pd.read_csv( os.path.join(here,"test_survived.csv"))

df_join1 = test_survived.join(result_81.set_index('PassengerId'),on='PassengerId', lsuffix='_test',rsuffix='_comp_81')

print(df_join1.loc[df_join1["Survived_test"] != df_join1["Survived_comp_81"],:])
df_join1.loc[df_join1["Survived_test"] != df_join1["Survived_comp_81"],:].to_csv(os.path.join(here,"check.csv"),index = False)
