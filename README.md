# SOEN-321-Adversarial-Attack-on-ML

## Goal
This project investigates how adversarial attacks can deceive machine learning models used in Intrusion Detection Systems(IDS).

Small changes in the input data can impact ML models, which demonstrates the importance of adversarial robustness.

## Objectives
- Train a Deep Neural Network (DNN) to classify network connections as either **normal** or **attack** based on the NSL-KDD dataset
- Apply data poisoning to train a separate DNN using **poisoned data** that will mislead the classifier
- Evaluate the difference in performance between our initial DNN and our poisoned DNN
- Link the robustness and limitations that ML models face to a cybersecurity context, more precisely network safety

## Dataset
The dataset used in this project is derived from the **NSL-KDD benchmark**:
 https://www.kaggle.com/datasets/hassan06/nslkdd?resource=download&select=KDDTest%2B.txt

The NSL-KDD benchmark is a widely used intrusion detection dataset (IDS), containing network connection records.

The dataset includes the following columns for each record:
`duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`,  
`hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`,  
`num_file_creations`, `num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login`,  
`count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`,  
`diff_srv_rate`, `srv_diff_host_rate`, `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`,  
`dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`,  
`dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate`, `label`, `difficulty`  

The **`label`** field specifies whether the connection was normal or represented as an attack type (e.g., neptune, ipsweep or warezclient).

- Training set: 125,000 samples 
- Testing set: 22,500 samples

## ML Models
For this project, a binary classification was performed, and all network traffic was regrouped under two classes "normal" or "attack".
Specific attack types are all part of the "attack" type.

Both DNN models were trained on records within the NSL-KDD dataset. All columns except the `label` and `difficulty` columns were used to train each model.

Then, the normal model and poisoned models' performances under **clean** and **adversarial** data was evaluated.

### Results
- Logistic Regression Accuracy on **clean DNN**: 0.7759
- Logistic Regression Accuracy on **poisoned DNN**: ---

## Adversarial Attack - Data Poisoning
Data poisoning is a type of attack in machine learning where an adversary manipulates or injects malicious data into a training dataset. This corrupts the model's learning process, which can cause it to make incorrect predictions or become biased when deployed.

## Technologies
- **Language**: Python
- **ML Frameworks**: scikit-learn, Tensorflow
- **Visualization**: 
- **Dataset**: NSL-KDD benchmark
- **Data Handling**: pandas, numpy

## Running the code
Install the minimal dependencies used in the notebook:
```bash
!pip install tensorflow numpy pandas scikit-learn
```

## Contributors 
| Name | Student ID | GitHub Username |
|------|------------|-----------------|
| Hiba Talbi | 40278717 | Yuioytffhio |
| Shayan Goldstein | 40229167 | shayanG7 |
| Hayk Petrosyan | 40310863 | zehayk |
