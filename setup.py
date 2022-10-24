from setuptools import setup

setup(
    name='speccaf',
    version='1.4.1',    
    description="Python implementation of a Spectral Continuum Anisotropic Fabric evolution model",
    url='https://github.com/danrichards678/SpecCAF',
    author='Daniel Richards',
    author_email='danrichards678@gmail.com',
    license='MIT',
    packages=['speccaf'],
    package_data={'speccaf':['data/*.npz']},
    install_requires=['numpy',
                      'scipy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)
