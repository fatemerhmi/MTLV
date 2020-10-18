# radiologyreportBESTproject
This is a project I'm doing for my thesis. 

## How to clone the repo
As this repo includes a sub module, the clone part is a bit different: 

Learn more about sub modules [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules). 

To clone the repo simply use:     
`git clone --recurse-submodules https://github.com/maktaf/radiologyreportBESTproject.git`

If you have already clone it without `--recurse-submodules`, this is how to add the submodules:
1. Clone the repo    
`git clone https://github.com/maktaf/radiologyreportBESTproject.git`
2. change the directory to the repo   
`cd radiologyreportBESTproject`
3. The submodule is the mimic-cxr directory, but empty.     
First run: ```git submodule init```      
This will initialize your local configuration file     

  Then: `git submodule update`      
  This will fetch all the data from `mimic-cxr` project and check out the appropriate commit listed in `radiologyreportBESTproject` project  

## Set up the environment
./setup.sh

## Run your experiments:
python3 src/main.py --help       
python3 src/main.py run --config src/mtl/config/openI_1layer.yml      
python3 src/main.py run --config src/mtl/config/openI_singlehead.yml --gpu-id 1

### MTL head configuration example options:
1. Any given set
```
head: 
  multi-task:
    heads: MultiLabelCLS
    type: givenset
    count: 4
    heads_index : [[0,1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19]]
```
2. KDE 
Bandwidth options: silverman, gridSearch, any given input(non negative int or float)
```
head: 
  multi-task:
    heads: MultiLabelCLS
    type: KDE
    bandwidth: 20
```

### loss configuration example options:
1. sum of head losses
```
loss:
  type: sumloss
```
2. weighted sum of head losses
Please note that you need to know how many heads you have. To figure it out. You can first run the same configuration with sumloss to check how many heads the algorithms have calculated for the run. 

3. average of head losses
```
loss:
  type: avgloss
```
4. weighted average of head losses


# Datasts:
1. [20newsgroup](http://qwone.com/~jason/20Newsgroups/)        
We used [bydate version](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) for the experiments in the thesis.       
The original version is also available in [sklearn](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)       

2. [Ohsumed]()            
The [original version](http://disi.unitn.eu/moschitti/corpora/ohsumed-all-docs.tar.gz) with 23 categories       
[Ohsmed O10](https://www.mat.unical.it/OlexSuite/Datasets/SampleDataSets-download.htm)       
[category description](http://disi.unitn.eu/moschitti/corpora/First-Level-Categories-of-Cardiovascular-Disease.txt)       

3. MIMIC-CXR
[Dataset Description](https://physionet.org/content/mimic-cxr/2.0.0/)
[Steps to Access the data](https://mimic.physionet.org/gettingstarted/access/)

4. Open-I         
[Dataset Description](https://openi.nlm.nih.gov/faq#collection)              
[Download](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz)          
If you identify openI dataset in the config file. It will automatically download and preprocess it for you.         
to see the details: `src/mtl/datasets/openI.py`    

5. Reuter           
The Reuters-21578 corpus consists of 21,578 news stories appeared on the Reuters newswire in 1987. However, the documents manually assigned to categories are only 12,902. These documents are classified across 135 categories. The ModAptè split subdivides the data set into a training and a test set of 9,603 and 3,299 documents, respectively. Once discarded all categories with no document in the test set, the remaining classification scheme is made of 90 categories (R90) and the remaining training set consists of 9,598 documents. Of the 90 categories of R90, we consider the standard subset consisting of the 10 most frequent (R10). [Source](https://www.mat.unical.it/OlexSuite/Datasets/SampleDataSets-about.htm)       
The [Original Version](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) with 135 categories
[Original Version Information files](http://kdd.ics.uci.edu/databases/reuters21578/README.txt)      
[Descirption of datast and 90 labels](https://martin-thoma.com/nlp-reuters/)    
[Routers R-10](https://www.mat.unical.it/OlexSuite/Datasets/data/R10/modApte.rar)            