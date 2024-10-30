from setuptools import setup, find_packages

setup(
    name="gate-manager",                         
    version="0.1.0",                             
    author="Chen Huang",                         
    author_email="chen.huang23@imperial.ac.uk",       
    description="A versatile device control class for managing Nanonis and Zurich devices",
    long_description=open("README.md").read(),   
    long_description_content_type="text/markdown",
    url="https://github.com/chenx820/gate-manager", 
    packages=find_packages(),                    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',                     # 支持的Python版本
    install_requires=[
        "nanonis_spm",
    ],
)
