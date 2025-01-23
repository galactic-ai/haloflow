import os, sys 


def train_npe_optuna(sim, obs, hr=12, gpu=True, mig=True, email="chhahn@princeton.edu"): 
    ''' train Neural Posterior Estimator for haloflow2 '''
    jname = "npe.%s.%s" % (sim, obs)
    ofile = "o/_NDE.%s.%s" % (sim, obs)

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --account=chhahn",
        ["#SBATCH --partition=standard", "#SBATCH --partition=gpu_standard"][gpu],
        "#SBATCH --nodes=1", 
        "#SBATCH --ntasks=16",
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', '#SBATCH --partition=mig'][mig], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=%s" % email,
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        # "conda activate sbi", 
        "cd haloflow",
        "source venv/bin/activate",
        "",
        "python /groups/chhahn/haloflow/bin/npe/npe.py %s %s" % (obs, sim), 
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


def validate_npe(sim, obs, hr=1, email="chhahn@princeton.edu"): 
    ''' validate Neural Posterior Estimator ensemble for haloflow2    '''
    jname = "valid.npe.%s.%s" % (sim, obs)
    ofile = "o/_valid.NDE.%s.%s" % (sim, obs)

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=%s" % email,
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        # "conda activate sbi", 
        "cd haloflow",
        "source venv/bin/activate",
        "",
        "python /groups/chhahn/haloflow/bin/npe/valid.py %s %s" % (obs, sim), 
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
    for sim in ['TNG50', 'TNG100', 'Eagle100', 'Simba100', 'TNG_ALL']: 
       train_npe_optuna(sim, 'mags', hr=8, gpu=True, mig=False, email='nikhilgaruda@arizona.edu') 
       train_npe_optuna(sim, 'mags_morph', hr=8, gpu=True, mig=False, email='nikhilgaruda@arizona.edu') 

    # for sim in ['TNG50', 'TNG100', 'Eagle100', 'Simba100', 'TNG_ALL']: 
    #     validate_npe(sim, 'mags', hr=1)
    #     validate_npe(sim, 'mags_morph', hr=1)

