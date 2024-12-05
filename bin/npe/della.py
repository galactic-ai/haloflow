'''

python script to deploy jobs on della-gpu


'''
import os, sys 


def train_npe_optuna(sim, obs, hr=12, gpu=True, mig=True): 
    ''' train Neural Posterior Estimator for haloflow2 
    '''
    jname = "npe.%s.%s" % (sim, obs)
    ofile = "o/_NDE.%s.%s" % (sim, obs)

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', '#SBATCH --partition=mig'][mig], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/haloflow/bin/npe/npe.py %s %s" % (obs, sim), 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


if __name__=="__main__": 
    #train_npe_optuna('TNG50', 'mags', hr=1, gpu=False, mig=False) 
    #train_npe_optuna('TNG50', 'mags_morph', hr=1, gpu=False, mig=False) 
    for sim in ['TNG100' 'Eagle100', 'Simba100']: 
        train_npe_optuna(sim, 'mags', hr=4, gpu=False, mig=False) 
        train_npe_optuna(sim, 'mags_morph', hr=4, gpu=False, mig=False) 
