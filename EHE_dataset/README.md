# Public Release of the EHE Dataset
The EHE dataset is published on paper:
**Skeleton-based human action evaluation using graph convolutional network for monitoring Alzheimer’s progression**
Bruce X.B. Yu, Yan Liu, Keith C.C. Chan, Qintai Yang, Xiaoying Wang, Pattern Recognition 2021 ([PDF](https://www.sciencedirect.com/science/article/pii/S003132032100282X))

# EHE Dataset Format
The naming format of the exercise repetitions are like "S##A##L##.skeleton", where S indicates subject id, A indicates exercise id, L indicates number of repetition.

# Prepare Data Evaluation Protocols
Two evaluation protocols are used: Cross-Subject and Random-Division.
Use file "polyu_elderlyhome_gendata.py" to generate the data for experiments.
*Note to set the skeleton data folder, the output folder, and the skeleton feature (i.e., position or orientation or both) at Line 129-133.

# Prepare Data for Pre-training
Use file "polyu_elderlyhome_gendata_cls.py" to generate the data to pre-train a GCN model.
