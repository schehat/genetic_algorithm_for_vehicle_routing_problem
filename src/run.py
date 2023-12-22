# Replace 'your_script.py' with the name of your Python file
python_file = 'main.py'

# Run the Python file
for i in range(5):
    exec(open(python_file).read())
