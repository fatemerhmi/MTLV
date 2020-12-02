import os
import yaml
import pandas as pd

MLRUNS = "./mlruns"
EXPERIMENTS = ["1","2","3","4"]

def store_experiment_results(experiment):
    df = pd.DataFrame(columns = ["experiment_name", "runName", "acc", "acc_std", "f1_macro", "f1_macro_std","f1_micro", "f1_micro_std"])

    PATH = f"{MLRUNS}/{experiment}"
    f = open(f"{PATH}/meta.yaml", 'r') 
    yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    experiment_name = yaml_cfg['name']
    
    f.close()
    for entry in os.listdir(PATH):
        if entry == "meta.yaml":
            continue
        RUN_PATH = f"{PATH}/{entry}/metrics"
        if os.path.exists(f"{RUN_PATH}/test.acc.mean"):
            f = open(f"{PATH}/{entry}/tags/mlflow.runName", 'r') 
            run_name = f.read()
            f.close()

            ff = open(f"{RUN_PATH}/test.acc.mean", 'r')
            test_acc_mean = round(float(ff.read().split()[1]),2)

            ff = open(f"{RUN_PATH}/test.acc.std", 'r')
            test_acc_std = round(float(ff.read().split()[1]),2)

            ff = open(f"{RUN_PATH}/test.f1_macro.mean", 'r')
            test_f1_macro_mean = round(float(ff.read().split()[1]),2)

            ff = open(f"{RUN_PATH}/test.f1_macro.std", 'r')
            test_f1_macro_std = round(float(ff.read().split()[1]),2)

            ff = open(f"{RUN_PATH}/test.f1_micro.mean", 'r')
            test_f1_micro_mean = round(float(ff.read().split()[1]),2)

            ff = open(f"{RUN_PATH}/test.f1_micro.std", 'r')
            test_f1_micro_std = round(float(ff.read().split()[1]),2)

            ff.close()
            #add to df
            df = df.append({"experiment_name":experiment_name, "runName":run_name, "acc":test_acc_mean, "acc_std":test_acc_std, "f1_macro":test_f1_macro_mean, "f1_macro_std":test_f1_macro_std, "f1_micro":test_f1_micro_mean, "f1_micro_std": test_f1_micro_std}, ignore_index=True)
    return df

writer = pd.ExcelWriter('./results.xlsx', engine='xlsxwriter')
for experiment in EXPERIMENTS:
    df = store_experiment_results(experiment)
    df.to_excel(writer, sheet_name=experiment)
writer.save()