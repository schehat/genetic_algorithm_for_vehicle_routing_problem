import subprocess

python_file = 'main.py'

# Run the Python file
for i in range(10):
    subprocess.run(['python', python_file])
