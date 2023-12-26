import subprocess

python_file = 'main.py'

# Run the Python file
for i in range(5):
    subprocess.run(['python', python_file])
