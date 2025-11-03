# SOEN-321-Adversarial-Attack-on-ML

## Goal
This project investigates how adversarial attacks can deceive machine learning models used in Intrusion Detection Systems(IDS).

Small changes in the input data can impact ML models, which demonstrates the importance of adversarial robustness.

To investigate this, we explored the vulnerability of models trained on network traffic data by implementing the Fast Gradient Sign Method (FGSM). 

## Objectives
- Classify network traffic as **normal** or **attack** using ML models 
- Train Logistic Regression 
- Apply FGSM to generate **adversarial samples** that will mislead the classifier, while making sure that every record remains valid 
- Evaluate how the model's accuracy changes before and after the attack  
- Link the robustness and limitations that ML models face in cybersecurity context, more precisely network safety. 

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
For this project, a binary classification was performed, all network traffic was regrouped under two classes "normal" or "attack".
Specific attack types are all part of the "attack" type.

Two baseline models were trained and compared:
- **Logistic Regression**
- **Random Forest**

After training, FGSM (Fast Gradient Sign Method) was applied to the Logistic Regression model, to generate adversarial examples.

Then, the model's performance under **clean** and **adversarial** data, was evaluated.

### Results
- Logistic Regression Accuracy on **clean set**: 0.7539
- Logistic Regression Accuracy on **adversarial set**: 0.2408

## Adversarial Attack - FGSM
The Fast Gradient Sign Method, is a white box attack that perturbs input features to increase the model's loss. It computed the gradient of the loss of the input, by combining a white box approach with a missclassification goal.

Even small changes can cause the classifier to missclassify network traffic, which demonstrates the model's vulnirability.

The FGSM, needs for a model to be differentiable to effectively perform the gradient and disturb the model. Hence why this attack was used on the trained Logistic Regression model, since LR is differentiable.

## Technologies
- **Language**: Python
- **ML Frameworks**: scikit-learn, ART (Adversarial Robustness Toolbox)
- **Visualization**: matplotlib, seaborn
- **Dataset**: NSL-KDD benchmark
- **Data Handling**: pandas, numpy

## Running the code
Install the minimal dependencies used in the notebook:
```bash
!pip install pandas numpy scikit-learn matplotlib seaborn -q
```

## Contributors 
| Name | Student ID | GitHub Username |
|------|------------|-----------------|
| Hiba Talbi | 40278717 | Yuioytffhio |
