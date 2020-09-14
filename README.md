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

# Run your experiments:
python3 src/main.py --help       
python3 src/main.py run --config src/mtl/config/openI_1layer.yml      


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

5. Reuter           
The [Original Version](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) with ... categories
[Descirption of datast and 90 labels](https://martin-thoma.com/nlp-reuters/)          
[Routers R-10](https://www.mat.unical.it/OlexSuite/Datasets/data/R10/modApte.rar)            