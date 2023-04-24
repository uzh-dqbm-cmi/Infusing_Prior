# ClinicalNLP-Workshop-ACL-2023
Official code implementation of the paper &lt;Boosting Radiology Report Generation by Infusing Comparison Prior>

Our code consists of two part: "Infusing_prior" and "Labeler". 

"Infusing_prior" includes the code of main model shown in figure 1 of the paper and "Labeler" is the code for rule-based labeler introduced in Sec. 3.1 in our paper.

Belows are the flow of process to run our code.

1. Run rule-based labeler to generate labeled file in csv. 

2. Save the labeled file in "Infusing_prior/data/" folder.

3. Run bash file (.sh) to train our main model.
