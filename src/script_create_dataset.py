import subprocess
import torch

def run_ppo_train_script(game, folder, create_mode_images = False):
    # Define the command and parameters as a list
    
    command = [
        'python', 'create_dataset_using_OCAtari.py',
        '-g', game,
        '-f', folder,
    ]
    if create_mode_images:
        command.append('--compute_root_images')

    # Execute the command without capturing the output, so it's displayed in the terminal
    result = subprocess.run(command, text=True)

    # Check if the execution was successful
    if result.returncode == 0:
        print(f"Execution successful for parameters: {command[4:]}")
    else:
        print(f"Execution failed for parameters: {command[4:]}")

if __name__ == "__main__":

    # Define different sets of parameters
    games = ["Pong", "Skiing", "Boxing"]
    folder = ["validation", "test"]

    for game in games:
        f = "test" # doesn't matter when creating mode images
        #run_ppo_train_script(game, f, create_mode_images = True)
        for f in folder:
            run_ppo_train_script(game, f, create_mode_images = False)
