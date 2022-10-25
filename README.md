# SpecCAF: Python implementation of a Spectral Continuum Anisotropic Fabric evolution model 

A spectral fabric evolution model for ice based on the theory of mixtures of continuous diversity (Faria, 2006; Placidi, 2010) that has been shown to reproduce experimentally observed fabrics (Richards et al. 2021). The model incorporates the effects of rigid-body rotation, basal-slip deformation, migration and rotational recrystallization, and requires velocity gradient and temperature as inputs. 

## Installation

`pip3 install speccaf`

Currently, matrices are built for L=4,6,8,12 and 20. If you wish to run the model with a different L value to this, you must first use the script matrixbuild.py to save pre-allocated arrays, and move this file to the packages local data directory.

## References

Faria, S.H. (2006). Creep and recrystallization of large polycrystalline masses. I. General continuum theory. https://doi.org/10.1098/rspa.2005.1610

Placidi, L., Greve, R., Seddik, H., Faria S.H., (2010) Continuum-mechanical, Anisotropic Flow model for polar ice masses, based on an anisotropic Flow Enhancement factor. https://doi.org/10.1007/s00161-009-0126-0

Richards, D.H.M,  Pegler, S.S, Piazolo, S., Harlen, O.G., (2021)
The evolution of ice fabrics: A continuum modelling approach validated against laboratory experiments https://doi.org/10.1016/j.epsl.2020.116718

