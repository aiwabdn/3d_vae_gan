from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Pillow>=2.7', 'Keras>=2.0']

if __name__ == '__main__':
    setup(
        name='ganapp',
        version='0.1',
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        include_package_data=True,
        description='3DWGAN')
