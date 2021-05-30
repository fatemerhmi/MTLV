The details of the two datasets used in this research. The split for the OHSUMED dataset is already provided with the dataset, while Open-I dataset splitting was performed using multi-label stratification method \cite{szymanski2017scikit}. During the experiments, the validation set is always 15\% of the training set. We used 20\% of OHSUMED dataset, which was extracted using stratified sampling~\cite{szymanski2017scikit} to save computational cost. The total, train and test are the count of unique documents in each set.


| dataset          | \#labels | Total  | Train | Test |
| -----------------|--------- | -------|-------|------|    
| Open-I           |   19     | 3,159  | 2,527     | 632  |
| OHSUMED          |   23     | 34,389 | 27,556    | 6,833  |
