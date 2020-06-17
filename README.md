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