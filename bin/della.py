'''

python script to deploy jobs on della-gpu


'''
import os, sys 

def make_data(snapshot): 
    '''
    '''
    jname = "data.%i" % snapshot
    ofile = "o/_data.%i" % snapshot 

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=01:59:59",
        "#SBATCH --export=ALL", 
        "#SBATCH --mem=16G",
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
        "python make_data.py %i" % snapshot, 
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


def train_NDE_optuna(obs, nf_model='maf', hr=12, gpu=True, mig=True): 
    ''' train NN compression of summary statistics
    '''
    jname = "NDE.%s.%s" % (obs, nf_model)
    ofile = "o/_NDE.%s.%s" % (obs, nf_model)

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
        "python nde.py %s %s" % (obs, nf_model), 
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

#for nsnap  in [91, 59, 63, 67, 72, 78, 84]: 
#    make_data(nsnap)

for i in range(10):
    train_NDE_optuna('mags', nf_model='maf', hr=4, gpu=False, mig=False) 
    train_NDE_optuna('mags_morph', nf_model='maf', hr=4, gpu=False, mig=False) 
    #train_NDE_optuna('mags_morph_satlum_all', nf_model='maf', hr=4, gpu=False, mig=False) 
    #train_NDE_optuna('mags_morph_satlum_all_rich_all', nf_model='maf', hr=4, gpu=False, mig=False) 
    train_NDE_optuna('mags_morph_satlum_mrlim', nf_model='maf', hr=4, gpu=False, mig=False) 
    train_NDE_optuna('mags_morph_satlum_mrliml_rich_mrlim', nf_model='maf', hr=4, gpu=False, mig=False) 
