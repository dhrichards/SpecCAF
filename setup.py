from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name='speccaf',
    version='1.4.4',    
    description="Python implementation of a Spectral Continuum Anisotropic Fabric evolution model",
    url='https://github.com/danrichards678/SpecCAF',
    author='Daniel Richards',
    author_email='danrichards678@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['speccaf'],
    package_data={'speccaf':['data/*.npz']},
    install_requires=['numpy',
                      'scipy',                     
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)
