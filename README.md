Replication files for [Chernozhukov, Newey, Quintas-Martínez and Syrgkanis (2021) "RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests"](https://arxiv.org/abs/2110.03031)

## Main Files

1. **RieszNet_IHDP.ipynb** runs the ATE MAE and coverage experiments based on IHDP semi-synthetic data using RieszNet. It takes the files in "data/IHDP" as inputs, and outputs in "results/IHDP/RieszNet." It produces a table "IHDP_MAE_NN.tex" and a histogram "IHDP_coverage_NN.pdf," which correspond to Table 1 Panel (a) and Figure 1 panel (a) in the paper. It also produces the ablation study results in Table 3.
2. **ForestRiesz_IHDP.ipynb** runs the ATE MAE and coverage experiments based on IHDP semi-synthetic data using ForestRiesz. It takes the files in "data/IHDP" as inputs, and outputs in "results/IHDP/ForestRiesz." It produces a table "IHDP_MAE_RF.tex" and a histogram "IHDP_coverage_RF.pdf," which correspond to Table 1 Panel (b) and Figure 1 panel (b) in the paper.
3. **RieszNet_BHP.ipynb** runs the average derivative experiments based on BHP semi-synthetic data using RieszNet. It takes the file in "data/BHP" as input, and outputs in "results/BHP/RieszNet." It produces a table collecting all results "res_avg_der_NN.tex" and, for each design, a histogram "all.pdf" in the corresponding sub-folder. Outputs "res_avg_der_NN.tex" and "true_f_compl_nonlin_conf/all.pdf" correspond to Table A1 and Figure 2 panel (a) of the paper, respectively. Table 2 panel (a) is the last row of Table A1.
4. **ForestRiesz_BHP.ipynb** runs the average derivative experiments based on BHP semi-synthetic data using ForestRiesz. It takes the file in "data/BHP" as input, and outputs in "results/BHP/ForestRiesz." It produces a table collecting all results "res_avg_der_RF.tex" and, for each design, "(method)_all.pdf" in the corresponding sub-folder, where (method) is a string detailing the type of cross-fitting (0 = none, 1 = simple, 2 = double) and whether multitasking is used (0 = No, 1 = Yes). Outputs "res_avg_der_NN.tex" and "true_f_compl_nonlin_conf/all.pdf" correspond to Table A2 and Figure 2 panel (b) of the paper, respectively. Table 2 panel (b) is the last row of Table A2, and the ablation results in Table 4 are the last five rows of Table A2. This file also produces the RF Plugin benchmark results in Table A3, which are stored in "plugin.tex".

## Utils Folder

1. **riesznet.py** contains the main class for RieszNet.
2. **forestriesz.py** contains the main class for ForestRiesz.
3. **moments.py** defines some moment functions to use with RieszNet (currently only ATE and avg_small_diff are used).
4. **NN_avgmom_sim** defines functions for experiments with semi-synthetic data that are used in **RieszNet_BHP.ipynb**.
5. **RF_avgmom_sim** defines functions for experiments with semi-synthetic data that are used in **ForestRiesz_BHP.ipynb**.
6. **ihdp_data.py** contains utils to load and format the IHDP data, and it is largely drawn from [Shi et el. (2019)'s replication code](https://github.com/claudiashi57/dragonnet).

## Data Folder

1. **IHDP** has two subfolders: **sim_data**, which contains 1000 semi-synthetic datasets based on IHDP that were generated using [Dorie (2016)'s `NPCI` R Package](https://github.com/vdorie/npci) under setting "A", and **sim_data_redraw_T**, hich contains 100 semi-synthetic datasets based on IHDP, also generated using `NPCI` under setting "A", but redrawing the treatment variable according to the propensity score setting "True". These are used for the ATE MAE and coverage experiments, respectively.
2. **BHP** contains the gasoline demand data from [Blundell et al. (2017)'s replication files](https://dataverse.harvard.edu/dataset.xhtml;jsessionid=ab284f8afb3805aad6f8c6b9ddca?persistentId=doi%3A10.7910%2FDVN%2F0YALNP&version=&q=&fileTypeGroupFacet=%22Data%22&fileAccess=&fileTag=&fileSortField=&fileSortOrder=), converted into csv. These are used to generate semi-synthetic data for the average derivative experiments.
