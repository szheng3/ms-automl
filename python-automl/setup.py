# Library imports
from setuptools import setup, find_packages

# Extract requirements from the requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
from setuptools import setup

setup(
    name='automl',
    version='1.0.0',
    packages=find_packages(),
    entry_points="""
    [console_scripts]
    whisper=wisper.models:audioToText
    """,
    url="https://github.com/szheng3/automl",
    license='',
    author='Shuai Zheng',
    author_email='szheng3@outlook.com',
    description='automl'
)
