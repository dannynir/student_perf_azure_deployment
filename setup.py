from setuptools import find_packages,setup
from typing import List

HYPEN_EDOT = '-e .'

def get_requirements(filepath:str)->List[str]:
    '''
    this function will return list of packages
    '''
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n","") for req in file_obj]

    if HYPEN_EDOT in requirements:
        requirements.remove(HYPEN_EDOT)
    
    return requirements


setup(
    name='mlproject',
    version= '0.0.1',
    author= 'tings',
    author_email= 'nirmal.chand3791@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)