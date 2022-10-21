# Predict_CAHD_by_ECG_on_UK_Biobank

This project contains 4 code files and a data set information file, which are briefly introduced as follows:

1. CAHD_merged_raw_signal_into_one_beat

  merged raw signal matrix [12,5000] to one beat[12,460] #460 can change
    
2. CAHD_machine_learning

   Traditional machine learning methods are used to predict CAHD
   
   
3. CAHD_merged_beat_model

   Deep learning method based on merged beat

4. CAHD_Sample_info.csv   
 All the sample information used in the experiment, the label  1 is the CAHD sample. For copyright reasons, please download the raw ECG file from the UK biobank. 

5. Supplement file.pdf
 Description of 34 clinical features and parameters of the Xgboost method.
