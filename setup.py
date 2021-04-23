from setuptools import setup, find_packages

setup(
    name="Hartree-Fock",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "quantum-systems @ git+https://github.com/Schoyen/quantum-systems.git",
    ],
)
