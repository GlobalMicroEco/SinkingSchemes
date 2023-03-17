# a simple numberic solution for the burgers' equation: du/dt + du^2/2dx = 0
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp

plt.style.use('_mpl-gallery')


class burger_solver():
    # Customize the f function that will be used in the second term of the equation.
    def __init__(self, ffunc):
        self.ffunc = ffunc

    # Central difference scheme
    def dudt_central(self, tt, u_bar):
        dx = 1/len(u_bar)
        f_bar = self.ffunc(u_bar)
        f_interface = (f_bar[1:] + f_bar[:-1]) / 2
        f_interface = np.insert(f_interface, 0, 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)
        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt

    # First order upwind scheme
    def dudt_firstorder_upwind(self, tt, u_bar):
        dx = 1/len(u_bar)
        f_bar = self.ffunc(u_bar)
        slope_interface = np.zeros(len(u_bar)-1)
        for ii in range(len(u_bar)-1):
            if u_bar[ii+1] != u_bar[ii]:
                slope_interface[ii] = (self.ffunc(u_bar[ii+1]) - self.ffunc(u_bar[ii])) / (u_bar[ii+1] - u_bar[ii])
            else:
                slope_interface[ii] = 0

        f_interface = f_bar[:-1] * (slope_interface > 0) +f_bar[1:] * (slope_interface <= 0)
        f_interface = np.insert(f_interface, 0, 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)
        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt

    # Third order upwind scheme or Quadratic Upstream Interpolation for Convective Kinematics (QUICK) scheme
    def dudt_QUICK(self, tt, u_bar):
        dx = 1/len(u_bar)
        f_bar = self.ffunc(u_bar)
        slope_interface = np.zeros(len(u_bar) - 1)
        f_interface = np.zeros(len(u_bar) - 1)
        f_bar_ext = np.zeros(len(u_bar) + 2)
        f_bar_ext[1:-1] = f_bar
        for ii in range(len(u_bar) - 1):
            if u_bar[ii + 1] != u_bar[ii]:
                slope_interface[ii] = (self.ffunc(u_bar[ii + 1]) - self.ffunc(u_bar[ii])) / (u_bar[ii + 1] - u_bar[ii])
            else:
                slope_interface[ii] = 0

            if slope_interface[ii] > 0:
                f_interface[ii] = -1/8 * f_bar_ext[ii] + 6/8 * f_bar_ext[ii+1] + 3/8 * f_bar_ext[ii+2]
            else:
                f_interface[ii] = -1/8 * f_bar_ext[ii+3] + 6/8 * f_bar_ext[ii+2] + 3/8 * f_bar_ext[ii+1]

        f_interface = np.insert(f_interface, 0, 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)
        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt

    # First order Godunov scheme + central difference, taking the average volumn as the left and right flux value.
    def dudt_Godunov_firstorder(self, tt, u_bar):
        dx = 1 / len(u_bar)
        f_bar = self.ffunc(u_bar)
        f_interface_max = np.maximum(f_bar[1:], f_bar[:-1])
        f_interface_min = np.minimum(f_bar[1:], f_bar[:-1])
        f_interface_min[np.where(np.logical_and(u_bar[1:] > 0, u_bar[:-1] < 0))  ] = 0
        f_interface = f_interface_max * (u_bar[:-1] > u_bar[1:]) + f_interface_min * (u_bar[:-1] <= u_bar[1:])

        f_interface = np.insert(f_interface, 0, 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)

        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt

    # Second order upwind Godunov scheme.

    def dudt_Godunov_upwind(self, tt, u_bar):
        dx = 1 / len(u_bar)
        f_bar = self.ffunc(u_bar)
        f_extro_bar = np.insert(f_bar, 0, 0)
        f_extro_bar = np.insert(f_extro_bar, len(f_extro_bar), 0)

        f_extro_left = np.zeros(len(u_bar)-1)
        f_extro_right = np.zeros(len(u_bar)-1)

        for ii in range(len(u_bar) - 1):
            f_extro_left[ii] = f_extro_bar[ii+1] + (f_extro_bar[ii+1] - f_extro_bar[ii]) / (dx) * (dx / 2)
            f_extro_right[ii] = f_extro_bar[ii + 2] - (f_extro_bar[ii + 3] - f_extro_bar[ii+2]) / (dx) * (dx / 2)

        f_interface_max = np.maximum(f_extro_left, f_extro_right)
        f_interface_min = np.minimum(f_extro_left, f_extro_right)
        f_interface_min[np.where(np.logical_and(u_bar[1:] > 0, u_bar[:-1] < 0))  ] = 0
        f_interface = f_interface_max * (u_bar[:-1] > u_bar[1:]) + f_interface_min * (u_bar[:-1] <= u_bar[1:])

        f_interface = np.insert(f_interface, 0, 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)

        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt


    # Second order Godunov scheme with Van Leer limitor.
    def dudt_Godunov_vllimitor(self, tt, u_bar):
        dx = 1 / len(u_bar)
        f_bar = self.ffunc(u_bar)
        f_extro_bar = np.insert(f_bar, 0, 0)
        f_extro_bar = np.insert(f_extro_bar, len(f_extro_bar), 0)

        f_extro_left = np.zeros(len(u_bar)-1)
        f_extro_right = np.zeros(len(u_bar)-1)

        for ii in range(len(u_bar) - 1):
            r_left = (f_extro_bar[ii+2] - f_extro_bar[ii+1]) / (f_extro_bar[ii+1] - f_extro_bar[ii]) if f_extro_bar[ii+1] != f_extro_bar[ii] else 0
            r_right = (f_extro_bar[ii+1] - f_extro_bar[ii+2]) / (f_extro_bar[ii+2] - f_extro_bar[ii+3]) if f_extro_bar[ii+2] != f_extro_bar[ii+1] else 0

            f_extro_left[ii] = f_extro_bar[ii+1] + (f_extro_bar[ii+1] - f_extro_bar[ii]) / 2 * self.vllimitor(r_left)
            f_extro_right[ii] = f_extro_bar[ii + 2] - (f_extro_bar[ii + 3] - f_extro_bar[ii+2]) / 2 * self.vllimitor(r_right)

        f_interface_max = np.maximum(f_extro_left, f_extro_right)
        f_interface_min = np.minimum(f_extro_left, f_extro_right)
        f_interface_min[np.where(np.logical_and(u_bar[1:] > 0, u_bar[:-1] < 0))  ] = 0
        f_interface = f_interface_max * (u_bar[:-1] > u_bar[1:]) + f_interface_min * (u_bar[:-1] <= u_bar[1:])

        f_interface = np.insert(f_interface, 0, 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)

        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt

    def vllimitor(self, r):
        if r == -1:
            phi = 0
        else:
            phi = 2*r / (1+r) if r>0 else 0
        return phi



    # Color gradient
    def colorFader(self, c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The space grid (x axis)
    x = np.linspace(0, 1, 100)
    # The initial concentration of u
    u_ini = np.sin(x * 2 * np.pi)

    # Define the f function
    def ffunc(u_bar):
        return np.power(u_bar, 2)/2
    # fig1, ax2 = plt.subplots()
    # ax2.plot(x, u_ini)
    # plt.show()

    # Initialize the solver
    bs = burger_solver(ffunc)
    # Use ODE45 to integrate over time while the solver have dealt with the integration over space
    sol1 = solve_ivp(bs.dudt_QUICK, [0,1], u_ini, t_eval = [i/100 for i in range(100)])

    # Color platte
    c1 = 'red'  # blue
    c2 = 'black'  # green
    n = sol1.y.shape[1]

    # plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    for nrow in range(sol1.y.shape[1]):
        ax.plot(x, sol1.y[:, nrow], linewidth=2.0, color=bs.colorFader(c1,c2,nrow/n))

    # ax.set(xlim=(0, 1), xticks=np.arange(0, 1),
    #        ylim=(-1, 1), yticks=np.arange(-1, 1))
    plt.title('The third order upwind scheme (QUICK)')
    # plt.savefig("QUICK.png", bbox_inches='tight')


