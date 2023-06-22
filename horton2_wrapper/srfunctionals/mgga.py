__all__ = ["xc_mgga_fxc", "xc_mgga_exc", "xc_mgga_vxc"]


def xc_mgga_exc(xc_name, rho, sigma, tau, lapl, mu):
    raise NotImplementedError


def xc_mgga_vxc(xc_name, rho, sigma, tau, lapl, mu):
    raise NotImplementedError


def xc_mgga_fxc(xc_name, rho, sigma, tau, lapl, mu):
    raise NotImplementedError
