import subprocess

def run_ppo_train_script(game, task):
    # Define the command and parameters as a list
    # python3 main.py --task eval_model_and_classifier --config configs/my_atari_pong_gpu.yaml 
    command = [
        'python', 'main.py',
        '--task', task,
        '--config', f'configs/my_atari_{game.lower()}_gpu.yaml',
    ]

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
    tasks = ["eval", "eval_classifier", "eval_model_and_classifier",]
    for game in games:
        for task in tasks:
            run_ppo_train_script(game, task)
