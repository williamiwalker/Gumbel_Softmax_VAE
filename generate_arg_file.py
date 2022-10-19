# This creates a json file that has a list of arguments that each job in a slurm job array will read
import json
import os



################################
# VQVAE ARGUMENTS
################################
SUBFOLDER_NAME = 'gumbel_softmax_2'
ARG_FILE_NAME = 'arguments_' + SUBFOLDER_NAME +'.json'
ARGUMENT_FILE = '/nfs/gatsbystor/williamw/gprpm_plots/arg_files/'+ARG_FILE_NAME
# ARGUMENT_FILE = '/home/william/mnt/gatsbystor/gprpm_plots/arg_files/'+ARG_FILE_NAME


COMMENTS = 'paired MNIST images data. Using gumbel softmax VAE'



mainArgs = {
'MAIN_FOLDER': SUBFOLDER_NAME,
'num_workers': 4,
'latent_dim': 1,
'temp': 1.0,
'learning_rate': 1e-3,
'batch_size': 100,
'categorical_dim': 10,
'epochs': 100,
'log_interval': 10,
'hard': False
}

num_jobs = 10


arguments = {}

job_index = 0
for indj in range(num_jobs):
    currDict = mainArgs.copy()
    currDict['model'] = 'gumbel_softmax'
    currDict['SUB_FOLDER'] = 'gumbel_softmax_' + str(indj)
    currDict['seed'] = int(indj + 1)
    currDict['COMMENTS'] = COMMENTS
    arguments[job_index] = currDict
    job_index += 1

print('sbatch --array=0-'+ str(job_index-1) + ' train_gumbel_softmax_VAE.sbatch')





if os.path.exists(ARGUMENT_FILE):
    print('overwrite')
    # raise Exception('You tryina overwrite a folder that already exists. Dont waste my time, sucka')

with open(ARGUMENT_FILE, 'w') as f:
    json.dump(arguments, f, indent=4)
#

