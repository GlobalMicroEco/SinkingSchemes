# a simple numberic solution for the burgers' equation: du/dt + du^2/2dx = 0
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp

plt.style.use('_mpl-gallery')


class burger_solver():
    # Central difference scheme
    def dudt_central(self, tt, u_bar):
        dx = 1/len(u_bar)
        f_bar = np.power(u_bar, 2)  / 2
        f_interface = (f_bar[1:] + f_bar[:-1]) / 2
        f_interface = np.insert(f_interface, 0 , 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)

        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt

    # First order upwind scheme
    def dudt_firstorder_upwind(self, tt, u_bar):
        dx = 1/len(u_bar)
        f_bar = np.power(u_bar, 2)  / 2
        slope_interface = (u_bar[1:] + u_bar[:-1]) / 2
        f_interface = f_bar[:-1] * (slope_interface > 0) +f_bar[1:] * (slope_interface <= 0)
        f_interface = np.insert(f_interface, 0 , 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)
        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt


    # First order Godunov scheme + central difference, taking the average volumn as the left and right flux value.
    def dudt_Godunov_firstorder(self, tt, u_bar):
        dx = 1 / len(u_bar)
        f_bar = np.power(u_bar, 2) / 2
        f_interface_max = np.maximum(f_bar[1:], f_bar[:-1])
        f_interface_min = np.minimum(f_bar[1:], f_bar[:-1])
        f_interface_min[np.where(np.logical_and(u_bar[1:] > 0, u_bar[:-1] < 0))  ] = 0
        f_interface = f_interface_max * (u_bar[:-1] > u_bar[1:]) + f_interface_min * (u_bar[:-1] <= u_bar[1:])

        f_interface = np.insert(f_interface, 0, 0)
        f_interface = np.insert(f_interface, len(f_interface), 0)

        du_bar_dt = (f_interface[:-1] - f_interface[1:]) / dx
        return du_bar_dt

    # Second order Godunov scheme.
    def dudt_Godunov_upwind(self, tt, u_bar):
        dx = 1 / len(u_bar)
        f_bar = np.power(u_bar, 2) / 2
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

    # Color gradient
    def colorFader(self, c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    u_ini = np.sin(x * 2 * np.pi)

    sol1 = solve_ivp(bs.dudt_firstorder_upwind, [0,1], u_ini, t_eval = [i/100 for i in range(100)])

    c1 = 'red'  # blue
    c2 = 'black'  # green
    n = sol1.y.shape[1]

    # plot
    fig, ax = plt.subplots()
    for nrow in range(sol1.y.shape[1]):
        ax.plot(x, sol1.y[:, nrow], linewidth=2.0, color=bs.colorFader(c1,c2,nrow/n))

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #        ylim=(-1, 1), yticks=np.arange(-1, 1))

    plt.show()


