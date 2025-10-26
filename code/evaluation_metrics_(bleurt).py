# Work on colab
# Commented out IPython magic to ensure Python compatibility.
# import pandas as pd
# df = pd.read_csv('filename.csv')

!git clone https://github.com/google-research/bleurt.git
# %cd bleurt
!pip install .

!wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
!unzip BLEURT-20.zip

from bleurt import score

def calculate_bleurt(candidate, reference):
  scorer = score.BleurtScorer(checkpoint = 'BLEURT-20')
  bleurt_score = scorer.score(references=[reference], candidates=[candidate])
  return bleurt_score
