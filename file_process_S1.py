import numpy as np

file = r'J:\Drago Guggiana Nilo\Data\DG_180816_a\2018_10_03\2\preProcessed.npz'
contents = np.load(file, allow_pickle=True)

data = contents['data'].item()
metadata = contents['metadata'].item()

# analyze the DG info
DG_data = data['DG']


print('yay')
