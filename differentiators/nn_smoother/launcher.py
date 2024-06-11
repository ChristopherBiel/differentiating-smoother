from differentiators.utils import generate_base_command, generate_run_commands, dict_permutations
import differentiators.nn_smoother.exp as exp

general_configs = {
    'project_name': ['BNNSweep_240530'],
    'seed': [0, 1],
    'num_traj': [8, 12],
    'sample_points': [48, 64, 96],
    'smoother_feature_size': [32],
    'smoother_hidden_layers': [2],
    'smoother_particles': [12],
    'smoother_training_steps': [4000, 8000],
    'smoother_weight_decay': [4e-4],
    'smoother_train_share': [0.8],
    'smoother_type': ['DeterministicEnsemble'],
    'dyn_feature_size': [48, 64, 96],
    'dyn_hidden_layers': [2],
    'dyn_particles': [6, 12],
    'dyn_training_steps': [4000, 16000],
    'dyn_weight_decay': [4e-3, 4e-4],
    'dyn_train_share': [0.8],
    'dyn_type': ['DeterministicEnsemble', 'DeterministicFSVGDEnsemble'],
    'logging_mode_wandb': [2],
    'x_src': ['smoother'],
}

def main():
    command_list = []
    flags_combinations = dict_permutations(general_configs)
    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)
    
    # submit the jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem = 4 * 1028)

if __name__ == '__main__':
    main()    
