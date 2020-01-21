# Load libraries
import pandas as pd


dnscat_df = pd.read_csv("./TRAININGS_DATA/lld_lab_dnscat_features_added.csv")
iodine_df = pd.read_csv("./TEST_DATA/lld_lab_iodine_features_added.csv")


vertical_stack = pd.concat([dnscat_df , iodine_df], axis=0)

print(vertical_stack.head)

vertical_stack.to_csv('lld_lab_tunnel_features_added.csv', index=False)
