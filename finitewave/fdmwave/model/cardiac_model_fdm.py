import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.fdmwave.stencil.stencil2D.isotropic_stencil_2d_fdm import IsotropicStencil2DFDM
from finitewave.fdmwave.stencil.stencil3D.isotropic_stencil_3d_fdm import IsotropicStencil3DFDM
from finitewave.fdmwave.stencil.stencil2D.asymmetric_stencil_2d_fdm import AsymmetricStencil2DFDM
from finitewave.fdmwave.stencil.stencil3D.asymmetric_stencil_3d_fdm import AsymmetricStencil3DFDM


class CardiacModelFDM(CardiacModel):
    """
    Base class for cardiac models using the finite difference method.
    """
    def __init__(self):
        """
        Initializes the CardiacModelFDM instance with default parameters.
        """
        super().__init__()

    def select_stencil(self, tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        tissue : CardiacTissue2D
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if tissue.fibers is not None:
            if tissue.mesh.ndim == 2:
                return AsymmetricStencil2DFDM()
            elif tissue.mesh.ndim == 3:
                return AsymmetricStencil3DFDM()

        if tissue.mesh.ndim == 2:
            return IsotropicStencil2DFDM()
        elif tissue.mesh.ndim == 3:
            return IsotropicStencil3DFDM()
        
        raise ValueError("Unsupported mesh dimension: " + str(tissue.mesh.ndim))






