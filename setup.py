try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
    import os

    def find_packages(where='.', exclude=()):
        packages = []
        for root, dirs, files in os.walk(where):
            if '__init__.py' in files:
                package = root.replace(os.path.sep, '.')
                if package.startswith('.' + os.path.sep):
                    package = package[2:]
                if package not in exclude:
                    packages.append(package)
        return packages

from typing import List

hyphen_e = '-e .'
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements= [req.replace('\n', '') for req in requirements]
        if hyphen_e in requirements:
            requirements.remove(hyphen_e)
    return requirements
setup(
    name='my_package',
    version='0.0.1',
    author='banwa',
    author_email='alaoabdulrahman64@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)