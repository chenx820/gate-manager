from setuptools import setup, find_packages

setup(
    name="gate-manager",                         
    version="2.0.0",                             
    author="Chen Huang",                         
    author_email="chen.huang23@imperial.ac.uk",       
    description="A package for controlling and managing gate voltages in quantum devices.",
    long_description=open("README.md").read(),   
    long_description_content_type="text/markdown",
    url="https://github.com/chenx820/gate-manager", 
    packages=find_packages(),                    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',                    
    install_requires=[
        "nanonis_spm",
    ],
)
