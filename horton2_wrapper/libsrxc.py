from __future__ import division, print_function

from horton import *

from horton2_wrapper.srfunctionals.gga import xc_gga_exc, xc_gga_vxc
from horton2_wrapper.srfunctionals.lda import xc_lda_exc, xc_lda_vxc
from horton2_wrapper.srfunctionals.mgga import xc_mgga_exc, xc_mgga_vxc

__all__ = [
    "RLibXCSRWrapper",
    "RLibXCSRLDA",
    "RLibXCSRGGA",
    "RLibXCSRMGGA",
]


class RLibXCSRWrapper(object):
    """
    Wrapper for short-range functional.
    """

    def __init__(self, name, mu=0.5):
        r"""
        Initialize object.

        Parameters
        ----------
        name: basestring
            The name of short-range xc functional.
        mu: float, default=0.5
            The coefficients of :math:`\frac{erf(\mu|r-r'|)}{|r-r'|}` .
        """
        self._name = name.lower()
        self._mu = mu

        if self._name.startswith("sr_lda"):
            self._family = 1
        elif self._name.startswith("sr_gga"):
            self._family = 2
        elif self._name.startswith("sr_mgga"):
            self._family = 32
        else:
            raise RuntimeError("Unknown short-range xc functional: {}".format(self._name))

    @property
    def name(self):
        """The name of the functional."""
        return self._name

    @property
    def mu(self):
        """The parameter for range separation."""
        return self._mu

    @property
    def family(self):
        return self._family

    # no doc
    def compute_gga_exc(
        self, ndarray_rho, ndarray_sigma, ndarray_zk
    ):  # real signature unknown; restored from __doc__
        """
        RLibXCWrapper.compute_gga_exc(self, ndarray rho, ndarray sigma, ndarray zk)
        Compute the GGA energy density.

                Parameters
                ----------
                rho : np.ndarray, shape=(npoint,)
                    The total electron density.
                sigma : np.ndarray, shape=(npoint,)
                    The reduced density gradient norm.
                zk : np.ndarray, shape=(npoint,), output
                    The energy density.
        """
        ret = xc_gga_exc(self._name, ndarray_rho, ndarray_sigma, mu=self._mu)
        ndarray_zk[:] = ret["zk"].flatten()

    def compute_gga_vxc(
        self, ndarray_rho, ndarray_sigma, ndarray_vrho, ndarray_vsigma
    ):  # real signature unknown; restored from __doc__
        """
        RLibXCWrapper.compute_gga_vxc(self, ndarray rho, ndarray sigma, ndarray vrho, ndarray vsigma)
        Compute the GGA functional derivatives.

                For every input `x`, a functional derivative is computed, `vx`, stored in an array
                with the same shape.

                Parameters
                ----------
                rho : np.ndarray, shape=(npoint,)
                    The total electron density.
                sigma : np.ndarray, shape=(npoint,)
                    The reduced density gradient norm.
                vrho : np.ndarray, shape=(npoint,), output
                    The LDA part of the potential.
                vsigma : np.ndarray, shape=(npoint,), output
                    The GGA part of the potential, i.e. derivatives of the density w.r.t. the
                    reduced gradient norm.
        """
        ret = xc_gga_vxc(self._name, ndarray_rho, ndarray_sigma, mu=self._mu)
        ndarray_vrho[:] = ret["vrho"].flatten()
        ndarray_vsigma[:] = ret["vsigma"].flatten()

    def compute_lda_exc(self, ndarray_rho, ndarray_zk):
        """
        RLibXCWrapper.compute_lda_exc(self, ndarray rho, ndarray zk)
        Compute the LDA energy density.

                Parameters
                ----------
                rho : np.ndarray, shape=(npoint,)
                    The total electron density.
                zk : np.ndarray, shape=(npoint,), output
                    The energy density.
        """
        ret = xc_lda_exc(self._name, ndarray_rho, mu=self._mu)
        ndarray_zk[:] = ret["zk"].flatten()

    def compute_lda_vxc(
        self, ndarray_rho, ndarray_vrho
    ):  # real signature unknown; restored from __doc__
        """
        RLibXCWrapper.compute_lda_vxc(self, ndarray rho, ndarray vrho)
        Compute the LDA potential.

                Parameters
                ----------
                rho : np.ndarray, shape=(npoint,)
                    The total electron density.
                vrho : np.ndarray, shape=(npoint,), output
                    The LDA potential.
        """
        ret = xc_lda_vxc(self._name, ndarray_rho, mu=self._mu)
        ndarray_vrho[:] = ret["vrho"].flatten()

    def compute_mgga_exc(
        self, ndarray_rho, ndarray_sigma, ndarray_lapl, ndarray_tau, ndarray_zk
    ):  # real signature unknown; restored from __doc__
        """
        RLibXCWrapper.compute_mgga_exc(self, ndarray rho, ndarray sigma, ndarray lapl, ndarray tau, ndarray zk)
        Compute the MGGA energy density.

                Parameters
                ----------
                rho : np.ndarray, shape=(npoint,)
                    The total electron density.
                sigma : np.ndarray, shape=(npoint,)
                    The reduced density gradient norm.
                lapl : np.ndarray, shape=(npoint,)
                    The laplacian of the density.
                tau : np.ndarray, shape=(npoint,)
                    The kinetic energy density.
                zk : np.ndarray, shape=(npoint,), output
                    The energy density.
        """
        ret = xc_mgga_exc(
            self._name, ndarray_rho, ndarray_sigma, ndarray_tau, ndarray_lapl, mu=self._mu
        )
        ndarray_zk[:] = ret["zk"].flatten()

    def compute_mgga_vxc(
        self,
        ndarray_rho,
        ndarray_sigma,
        ndarray_lapl,
        ndarray_tau,
        ndarray_vrho,
        ndarray_vsigma,
        ndarray_vlapl,
        ndarray_vtau,
    ):
        """
        RLibXCWrapper.compute_mgga_vxc(self, ndarray rho, ndarray sigma, ndarray lapl, ndarray tau, ndarray vrho, ndarray vsigma, ndarray vlapl, ndarray vtau)
        Compute the MGGA functional derivatives.

                For every input `x`, a functional derivative is computed, `vx`, stored in an array
                with the same shape.

                Parameters
                ----------
                rho : np.ndarray, shape=(npoint,)
                    The total electron density.
                sigma : np.ndarray, shape=(npoint,)
                    The reduced density gradient norm.
                lapl : np.ndarray, shape=(npoint,)
                    The laplacian of the density.
                tau : np.ndarray, shape=(npoint,)
                    The kinetic energy density.
                vrho : np.ndarray, shape=(npoint,)
                    The derivative of the energy w.r.t. the electron density.
                vsigma : np.ndarray, shape=(npoint,)
                    The derivative of the energy w.r.t. the reduced density gradient norm.
                vlapl : np.ndarray, shape=(npoint,)
                    The derivative of the energy w.r.t. the laplacian of the density.
                vtau : np.ndarray, shape=(npoint,)
                    The derivative of the energy w.r.t. the kinetic energy density.
        """
        ret = xc_mgga_vxc(
            self._name, ndarray_rho, ndarray_sigma, ndarray_tau, ndarray_lapl, mu=self._mu
        )
        ndarray_vrho[:] = ret["vrho"].flatten()
        ndarray_vsigma[:] = ret["vsigma"].flatten()
        ndarray_vtau[:] = ret["vtau"].flatten()
        ndarray_vlapl[:] = ret["vlapl"].flatten()


class RLibXCSRLDA(GridObservable):
    df_level = DF_LEVEL_LDA
    prefix = "sr_lda"
    LibXCWrapper = RLibXCSRWrapper

    def __init__(self, name, mu):
        """Initialize a LibXCEnergy instance.

        Parameters
        ----------
        name : str
            The name of the functional in LibXC, without the ``lda_``, ``gga_`` or
            ``hyb_gga_`` prefix. (The type of functional is determined by the subclass.)
        """
        name = "%s_%s" % (self.prefix, name)
        self._name = name
        self._libxc_wrapper = self.LibXCWrapper(name, mu)
        GridObservable.__init__(self, "libxc_%s" % name)

    @timer.with_section("LDA edens")
    @doc_inherit(LibXCEnergy)
    def compute_energy(self, cache, grid):
        # LibXC expects the following input:
        #   - total density
        # LibXC computes:
        #   - the energy density per electron.
        rho_full = cache["rho_full"]
        edens, new = cache.load("edens_libxc_%s_full" % self._name, alloc=grid.size)
        if new:
            self._libxc_wrapper.compute_lda_exc(rho_full, edens)
        return grid.integrate(edens, rho_full)

    @timer.with_section("LDA pot")
    @doc_inherit(LibXCEnergy)
    def add_pot(self, cache, grid, pots_alpha):
        # LibXC expects the following input:
        #   - total density
        # LibXC computes:
        #   - the potential for the alpha electrons.
        pot, new = cache.load("pot_libxc_%s_alpha" % self._name, alloc=grid.size)
        if new:
            self._libxc_wrapper.compute_lda_vxc(cache["rho_full"], pot)
        pots_alpha[:, 0] += pot

    # TODO: fixme
    # @timer.with_section('LDA dot')
    # @doc_inherit(LibXCEnergy)
    # def add_dot(self, cache, grid, dots_alpha):
    #     # LibXC expects the following input:
    #     #   - total density
    #     # This method also uses
    #     #   - change in total density
    #     # LibXC computes:
    #     #   - the diagonal second order derivative of the energy towards the
    #     #     density
    #     kernel, new = cache.load('kernel_libxc_%s_alpha' % self._name, alloc=grid.size)
    #     if new:
    #         self._libxc_wrapper.compute_lda_fxc(cache['rho_full'], kernel)
    #     dots_alpha[:, 0] += kernel * cache['delta_rho_full']


class RLibXCSRGGA(GridObservable):
    """Base class for LibXC functionals."""

    df_level = DF_LEVEL_GGA
    prefix = "sr_gga"
    LibXCWrapper = RLibXCSRWrapper

    def __init__(self, name, mu):
        """Initialize a LibXCEnergy instance.

        Parameters
        ----------
        name : str
            The name of the functional in LibXC, without the ``lda_``, ``gga_`` or
            ``hyb_gga_`` prefix. (The type of functional is determined by the subclass.)
        """
        name = "%s_%s" % (self.prefix, name)
        self._name = name
        self._libxc_wrapper = self.LibXCWrapper(name, mu)
        GridObservable.__init__(self, "libxc_%s" % name)

    @timer.with_section("GGA edens")
    @doc_inherit(LibXCEnergy)
    def compute_energy(self, cache, grid):
        # LibXC expects the following input:
        #   - total density
        #   - norm squared of the gradient of the total density
        # LibXC computes:
        #   - energy density per electron
        rho_full = cache["rho_full"]
        edens, new = cache.load("edens_libxc_%s_full" % self._name, alloc=grid.size)
        if new:
            sigma_full = cache["sigma_full"]
            self._libxc_wrapper.compute_gga_exc(rho_full, sigma_full, edens)
        return grid.integrate(edens, rho_full)

    def _compute_dpot_spot(self, cache, grid):
        """Helper function to compute potential resutls with LibXC.

        This is needed for add_pot and add_dot.

        Parameters
        ----------
        cache : Cache
            Used to share intermediate results.
        grid : IntGrid
            A numerical integration grid.
        """
        dpot, newd = cache.load("dpot_libxc_%s_alpha" % self._name, alloc=grid.size)
        spot, news = cache.load("spot_libxc_%s_alpha" % self._name, alloc=grid.size)
        if newd or news:
            rho_full = cache["rho_full"]
            sigma_full = cache["sigma_full"]
            self._libxc_wrapper.compute_gga_vxc(rho_full, sigma_full, dpot, spot)
        return dpot, spot

    @timer.with_section("GGA pot")
    @doc_inherit(LibXCEnergy)
    def add_pot(self, cache, grid, pots_alpha):
        # LibXC expects the following input:
        #   - total density
        #   - norm squared of the gradient of the total density
        # LibXC computes:
        #   - the derivative of the energy towards the alpha density.
        #   - the derivative of the energy towards the norm squared of the alpha density.
        dpot, spot = self._compute_dpot_spot(cache, grid)

        # Chain rule: convert derivative toward sigma into a derivative toward
        # the gradients.
        my_gga_pot_alpha, new = cache.load(
            "gga_pot_libxc_%s_alpha" % self._name, alloc=(grid.size, 4)
        )
        if new:
            my_gga_pot_alpha[:, 0] = dpot
            grad_rho = cache["grad_rho_full"]
            my_gga_pot_alpha[:, 1:4] = grad_rho * spot.reshape(-1, 1)
            my_gga_pot_alpha[:, 1:4] *= 2

        # Add to the output argument
        pots_alpha[:, :4] += my_gga_pot_alpha


class RLibXCSRMGGA(GridObservable):
    """Base class for LibXC functionals."""

    df_level = DF_LEVEL_MGGA
    prefix = "sr_mgga"
    LibXCWrapper = RLibXCSRWrapper

    def __init__(self, name, mu):
        """Initialize a LibXCEnergy instance.

        Parameters
        ----------
        name : str
            The name of the functional in LibXC, without the ``lda_``, ``gga_`` or
            ``hyb_gga_`` prefix. (The type of functional is determined by the subclass.)
        """
        name = "%s_%s" % (self.prefix, name)
        self._name = name
        self._libxc_wrapper = self.LibXCWrapper(name, mu)
        GridObservable.__init__(self, "libxc_%s" % name)

    @timer.with_section("MGGA edens")
    @doc_inherit(LibXCEnergy)
    def compute_energy(self, cache, grid):
        # LibXC expects the following input:
        #   - total density
        #   - norm squared of the gradient of the total density
        #   - the laplacian of the density
        #   - the kinetic energy density
        # LibXC computes:
        #   - energy density per electron
        rho_full = cache["rho_full"]
        edens, new = cache.load("edens_libxc_%s_full" % self._name, alloc=grid.size)
        if new:
            sigma_full = cache["sigma_full"]
            lapl_full = cache["lapl_full"]
            tau_full = cache["tau_full"]
            self._libxc_wrapper.compute_mgga_exc(rho_full, sigma_full, lapl_full, tau_full, edens)
        return grid.integrate(edens, rho_full)

    @timer.with_section("MGGA pot")
    @doc_inherit(LibXCEnergy)
    def add_pot(self, cache, grid, pots_alpha):
        # LibXC expects the following input:
        #   - total density
        #   - norm squared of the gradient of the total density
        #   - the laplacian of the density
        #   - the kinetic energy density
        # LibXC computes:
        #   - the derivative of the energy towards the alpha density.
        #   - the derivative of the energy towards the norm squared of the alpha density.
        #   - the derivative of the energy towards the laplacian of the density
        #   - the derivative of the energy towards the kinetic energy density
        dpot, newd = cache.load("dpot_libxc_%s_alpha" % self._name, alloc=grid.size)
        spot, news = cache.load("spot_libxc_%s_alpha" % self._name, alloc=grid.size)
        lpot, news = cache.load("lpot_libxc_%s_alpha" % self._name, alloc=grid.size)
        tpot, news = cache.load("tpot_libxc_%s_alpha" % self._name, alloc=grid.size)
        if newd or news:
            rho_full = cache["rho_full"]
            sigma_full = cache["sigma_full"]
            lapl_full = cache["lapl_full"]
            tau_full = cache["tau_full"]
            self._libxc_wrapper.compute_mgga_vxc(
                rho_full, sigma_full, lapl_full, tau_full, dpot, spot, lpot, tpot
            )

        # Chain rule: convert derivative toward sigma into a derivative toward
        # the gradients.
        my_mgga_pot_alpha, new = cache.load(
            "mgga_pot_libxc_%s_alpha" % self._name, alloc=(grid.size, 6)
        )
        if new:
            my_mgga_pot_alpha[:, 0] = dpot
            grad_rho = cache["grad_rho_full"]
            my_mgga_pot_alpha[:, 1:4] = grad_rho * spot.reshape(-1, 1)
            my_mgga_pot_alpha[:, 1:4] *= 2
            my_mgga_pot_alpha[:, 4] = lpot
            my_mgga_pot_alpha[:, 5] = tpot

        # Add to the output argument
        pots_alpha[:, :6] += my_mgga_pot_alpha
