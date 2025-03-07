import os
import optuna
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from sbi import inference as Inference
from sbi import utils as Ut

class NPEOptunaTraining:
    # USAGE:
    # npe = NPEOptunaTraining(y, x, n_trials, study_name, output_dir, n_jobs, device)
    # study = npe()
    def __init__(self, y, x, 
                n_trials, 
                study_name,
                output_dir,
                n_jobs,
                device,
                ):
       
        self.y = y
        self.x = x

        self.n_trials = n_trials
        self.device = device
        self.study_name = study_name
        self.output_dir = output_dir
        self.n_jobs = n_jobs

        self.n_startup_trials = 20
        self.n_blocks_min = 2 
        self.n_blocks_max = 5
        self.n_transf_min = 2
        self.n_transf_max = 5
        self.n_hidden_min = 32
        self.n_hidden_max = 128
        self.n_lr_min = 5e-6
        self.n_lr_max = 1e-3

        self.storage = self._create_storage()
        self.prior = self._set_prior()
    
    def _create_storage(self):
        if not os.path.isdir(os.path.join(self.output_dir, self.study_name)):
            os.system('mkdir %s' % os.path.join(self.output_dir, self.study_name))
        return 'sqlite:///%s/%s/%s.db' % (self.output_dir, self.study_name, self.study_name)
    
    def _set_prior(self):
        lower_bounds = torch.tensor([8., 8.])
        upper_bounds = torch.tensor([14., 15.])
        return Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=self.device)
    
    def objective(self, trial):
        n_blocks = trial.suggest_int("n_blocks", self.n_blocks_min, self.n_blocks_max)
        n_transf = trial.suggest_int("n_transf", self.n_transf_min,  self.n_transf_max)
        n_hidden = trial.suggest_int("n_hidden", self.n_hidden_min, self.n_hidden_max, log=True)
        lr = trial.suggest_float("lr", self.n_lr_min, self.n_lr_max, log=True)
        neural_posterior = Ut.posterior_nn('maf', 
                hidden_features=n_hidden, 
                num_transforms=n_transf, 
                num_blocks=n_blocks, 
                use_batch_norm=True)
        
        anpe = Inference.SNPE(prior=self.prior,
                density_estimator=neural_posterior,
                device=self.device, 
                summary_writer=SummaryWriter('%s/%s/%s.%i' % 
                    (self.output_dir, self.study_name, self.study_name, trial.number))
        )
        anpe.append_simulations(
                torch.tensor(self.y, dtype=torch.float32).to(self.device), 
                torch.tensor(self.x, dtype=torch.float32).to(self.device))
        p_theta_x_est = anpe.train(
                training_batch_size=50,
                learning_rate=lr, 
                show_train_summary=True)
        qphi = anpe.build_posterior(p_theta_x_est)
        fqphi = os.path.join(self.output_dir, self.study_name, '%s.%i.pt' % (self.study_name, trial.number))
        torch.save(qphi, fqphi)
        best_valid_log_prob = anpe._summary['best_validation_log_prob'][0]
        return -1*best_valid_log_prob
    
    def run(self):
        sampler = optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(study_name=self.study_name, storage=self.storage, sampler=sampler, directions=['minimize'], load_if_exists=True)
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        return study
    
    def __call__(self):
        return self.run()