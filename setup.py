from setuptools import setup, find_packages

required = [
     'bsm @ git+https://github.com/lasgroup/bayesian_statistical_models.git'
]

extras = {}
setup(
    name='diff_smoothers',
    version='0.0.1',
    description='Library of methods for differentiating and smoothing state estimates',
    author='Christopher Biel',
    author_email='cbiel01@ethz.ch',
    license="MIT",
    packages=find_packages(include=['diff_smoothers', 'diff_smoothers.*']),
    python_requires='>=3.11',
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)