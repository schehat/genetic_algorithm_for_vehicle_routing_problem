import subprocess

python_file = 'main.py'

# Run the Python file
for i in range(3):
    subprocess.run(['python', python_file])
