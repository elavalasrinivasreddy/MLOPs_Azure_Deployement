from setuptools import setup, find_packages

setup(
    name="gem-stone-price-pred",
    version='0.0.3',
    author="Elavala",
    author_email="elavalasrinivasreddy@gmail.com",
    install_requires=["scikit-learn","pandas"],
    packages=find_packages()
)