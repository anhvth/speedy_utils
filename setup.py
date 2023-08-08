from setuptools import setup, find_packages

setup(
    name='speedy',
    version='0.1.0',
    description='Fast and easy-to-use package for data science',
    author='AnhVTH',
    author_email='anhvth.226@gmail.com',
    url='https://github.com/anhvth/speedy',
    packages=find_packages(),  # Automatically find all packages in the directory
    install_requires=[  # List any dependencies your package requires
        'numpy',
        'requests',
    ],
    classifiers=[  # Provide information about your package for PyPI
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
