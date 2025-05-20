import os, sys

def slurm_sweep(n_cores, obs, dann_sim, sim, hr=12, gpu=True):
    """
    WandB sweep for NPE training with DANN.
    """
    jname = f"npe.dann.{str(n_cores)}_m{dann_sim}_{sim}_{obs}"
    ofile = f"o/_{jname}"

    script = '\n'.join([
        '#!/bin/bash',
        f'#SBATCH -J {jname}',
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={n_cores}",
        "#SBATCH --mem-per-cpu=6GB",
        "#SBATCH --account=chhahn",
        ["#SBATCH --partition=standard", "#SBATCH --partition=gpu_standard"][gpu],
        ['', "#SBATCH --gres=gpu:1"][gpu],
        f"#SBATCH --time={str(hr-1).zfill(2)}:59:59",
        "#SBATCH --export=ALL",
        f"#SBATCH --output={ofile}",
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=nikhilgaruda@arizona.edu",
        "",
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc",
        "cd haloflow",
        "source venv/bin/activate",
        "",
        "cd bin/dann/",
        "",
        f"python npe_dann.py {obs} {sim} {dann_sim}",
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

if __name__ == "__main__":
    all_sims = ['TNG_ALL', 'Eagle100', 'Simba100']
    # all_obs = ['mags', 'mags_morph', 'mags_morph_extra']
    all_obs = ['mags_morph_extra']

    for obs in all_obs:
        for sim in all_sims:
            for dann_sim in all_sims:
                if sim == dann_sim:
                    continue
                print(f"Submitting job for obs: {obs}, sim: {sim}, dann_sim: {dann_sim}")
                slurm_sweep(n_cores=8, obs=obs, dann_sim=dann_sim, sim=sim, hr=12, gpu=True)


