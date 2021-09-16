# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:11:50 2021

@author: Smegn
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AB_tests", # Replace with your own username
    version="0.0.1",
    author="Smegnsh D",
    author_email="smegnshdem@gmail.com",
    description="Sequential, Classic and ML a/b testing",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/smegnshd/AdSmart_campaign",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)