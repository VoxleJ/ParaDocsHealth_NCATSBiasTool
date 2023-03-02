# A Free-Text Linguistic Feature Extraction Tool to Detect & Mitigate Provider Implicit Bias Associated with Clinical Decision Making for Hospitalized Patients with Diabetic Ketoacidosis  

Dickson T. Chen, MS; Vibhav Jha, MS; Dr. Matt Segar, MD; Dr. Dhini Nasution, MD, MAP; Dr. Omar Mohtar, MD, PhD 

## Abstract
This work proposes a tool that employs traditional ML methods such as named entity recognition, sentiment analysis and random forest classification to detect and minimize implicit bias in de-identified free-text clinical discharge notes from the MIMIC IV V2.2 dataset. By extracting biased linguistic features connected to clinical decision making and performing context based negative sentiment analysis, our tool enables providers to detect specific clinical events where bias was introduced with respect to the patient’s entire care journey. Board-certified physicians blindly labeled 526 free-text clinical discharge notes of patients with diabetic ketoacidosis (DKA), assigning a binary bias classification based on clinical actions that deviated from standard medical practices to treat DKA and/or its associated symptoms. Upon detection of biased linguistic features, the tool will alert providers real-time, paraphrase, and neutralize the negative sentiment by suggesting neutral descriptors to empower confident clinical decision making for optimal patient outcomes. Data-driven insights from our tool can assist providers and healthcare administrators alike to bring about behavioral change to guard against bias propagation, promoting trust through explainability of AI/ML generated results and improved patient outcomes.   

## Installation Instructions
Instructions txt are located in the Code folder. Navigate to the Code folder and install requirements.txt. 
