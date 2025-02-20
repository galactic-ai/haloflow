import os, sys

def slurm_sweep(n_cores, hr=12, gpu=True, mig=True):
    """
    WandB sweep for DANN finetuning.
    """
    jname = f"sweep.dann.{str(n_cores)}"
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
        ['', '#SBATCH --partition=mig'][mig],
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
        #"wandb sweep wandb.yaml --project dann_sweep > sweep_id.log",
        "sweep_id=nikhil0504/dann_sweep/uh5lo65f",
        f"""
        for i in $(seq 1 {n_cores}); do
            wandb agent $sweep_id &
            sleep 2
        done
        """,
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
    slurm_sweep(n_cores=24, hr=12, gpu=True, mig=False)
