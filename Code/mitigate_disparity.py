# -*- coding: utf-8 -*-
"""
@author: vjha1
"""
#mitigate disparity just takes the output from measure_disparity, and paraphrases the biased phrase
#this is set as an example for a single biased note

from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")
#pip install protobuf==3.20.* for some reason need to downgrade

def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)


parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)


#%% Text input with Parrot
import pandas as pd

data = pd.read_csv('Labeled/KA_btwn_2_5_Labeled_Combined.csv')
phrases = data.crucial_statements.iloc[1]
#this is set up to keep a human in the loop
#phrases = ['At the end of ___ it was unclear where the pt was getting her insulin from and an A1c was 8.9.', 'The pt states that she had been taking her insulin as scheduled for the past few days but she is unclear as to the dose and type of medication she takes.', 'She denies nausea abdominal pain vomiting diarrhea chest pain sob.', 'She was aggressively hydrated with ___ and placed on an insulin drip per ___ protocol.', 'Per collateral information from neighbors and friends she has had increasing difficulty in the past few months with increasing confusion.', 'She even was noted to have gone to the wrong apartment building and was very argumentative with the concierge staff although the status of her acute illness was unknown at this time.', 'It is unclear if the patients confusion due to her acute illness might have contributed or if her suspected underlying congnitive issues may have instead contributed.', 'You also need to take insulin everyday to avoid coming back to the hospital for ketoacidosis.']

# phrases = ['The patient reports that over the past several weeks he has become lazy with his insulin regimen at home.', 'He estimates that he has missed approximately ___ of his insulin doses over this time frame.', 'He endorses feeling tired and sluggish though notes he is not far from his baseline.', 'FSBS was 71 @1600 FSBS was 150 @1630 and FSBS was 174 @1700.This morning he feels well and denies complaints.', 'He tolerated po dinner last night.', 'Also followed by ___ who did not recommend changing his home regimen.', 'He was counseled on the need to follow a healthier diet and stop smoking.', 'In other words your blood sugar levels were critically high causing a form of shock to your body.', 'We understand this is a difficult schedule to follow however your long-term health depends on this.']

phrases = ['He denies any symptoms including nausea vomiting diarrhea abdominal pain SOB URT symptoms recent febrile illness or any other complaints.', 'He denies recent dietary indiscretions and has been taking his insulin regularly in recent days.', 'He does however admit to poor adherence over all during the past year at around 85% to his assessment not regularly taking FSBS missing insulin shots and occasional dietary indiscretions.', 'During his admission he had no complaints and he tolerated PO nutrition and hydration well.', 'Medications on Admission:- ___ humalog 40 q AM- ___ humalog 20 q ___- Humalog SS- Lantus 30 units ___\n Discharge Medications:We made no changes to your medication.', 'We made no changes to your diabetes regimen and you should continue following ___ recommendations.', 'Prior to your discharge following lunch you had an elevated blood sugar - we re-checked your labs which showed no evidence of DKA.', 'We strongly encourage you to stop smoking as this extremely bad for your health.']


for phrase in phrases:
  print("-"*100)
  print("Input_phrase: ", phrase)
  print("-"*100)
  para_phrases = parrot.augment(input_phrase=phrase)
  for para_phrase in para_phrases:
   print(para_phrase)
      