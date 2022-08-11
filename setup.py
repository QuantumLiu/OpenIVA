from setuptools import setup, find_packages

def req_file(filename):
    with open( filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]

setup(
    name='openiva',
    version='0.0.1',
    author='QuantumLiu',
    author_email='quantumliu@pku.edu.cn',
    url='https://github.com/QuantumLiu',
    description='An end-to-end intelligent vision analytics development toolkit based on different inference backend',
    packages=find_packages(),
    install_requires=[req_file("requirements.txt")],
    
)