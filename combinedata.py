# Load libraries
import pandas as pd


dnscat_df = pd.read_csv("./TRAININGS_DATA/lld_lab_dnscat_features_added.csv")
iodine_df = pd.read_csv("./TEST_DATA/lld_lab_iodine_features_added.csv")


tunnel_df = pd.concat([dnscat_df , iodine_df], axis=0)

print(tunnel_df.head)
print("Shuffel data")
tunnel_df.sample(frac=1).reset_index(drop=True)

print("Split data")
test = tunnel_df.sample(frac=0.1,random_state=200)
train = tunnel_df.drop(test.index)
print("Write data..")
train.to_csv('./TRAININGS_DATA/lld_lab_tunnel_features_added_1.csv', index=False)
test.to_csv('./TEST_DATA/lld_lab_tunnel_features_added_1.csv', index=False)
