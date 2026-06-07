from setuptools import find_packages,setup
from typing import List



def get_requirements(file_path:str)->List[str]:
    re=[]
    with open(file_path) as file_obj:
        re=file_obj.readlines()
        re=[i.replace("\n","") for i in re]

        if "-e ." in re:
            re.remove("-e .")


setup(name="ml_project",
       version="0.0.1",
       author="ADYASHA",
       packages=find_packages(),
       install_requires=get_requirements("requirements.txt")
       )