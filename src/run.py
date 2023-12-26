import subprocess

python_file = 'main.py'

# Run the Python file
for i in range(4):
    subprocess.run(['python', python_file])
