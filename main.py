import numpy as np
import math
import matplotlib.pyplot as plt

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""


def getAcc(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    return a


def getEnergy(pos, vel, mass, G):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum(mass * vel ** 2))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE
class Body():
    def __init__(self, mass, position, velocity, color='blue'):
        # Rescalers for the units
        m_r = 1.0e24 #mass rescaler
        r_r = 1.0e8  #length rescaler
        v_r = 8.169638912e2  #velocity rescaler
        self.mass = np.array([mass/m_r])
        self.position = np.array([position[0], position[1], 0])/r_r
        self.velocity = np.array([velocity[0], velocity[1], 0])/v_r
        self.color = color

def polar(r, theta, normal=0):
    if normal:
        return np.array([-r * math.sin(theta), r * math.cos(theta)])
    return np.array([r * math.cos(theta), r * math.sin(theta)])

def main():
    """ N-body simulation """

    # Simulation parameters
    t = 0  # current time of the simulation
    tEnd = 236.287  # time at which simulation ends
    dt = 0.1  # timestep
    softening = 0  # softening length
    G = 1  # Newton's Gravitational Constant
    plotRealTime = False  # switch on for plotting as the simulation goes along


    # Visual Parameters
    log_base = 1.01
    min_size = 50
    square_range = 1600
    looking_range = [[-square_range, square_range], [-square_range, square_range]]      #The range you'll be looking at away from the reference [[xmin, xmax], [ymin, ymax]]
    trail_len = 0
    trail_color = 'gray'


    sun_mass = 1.9891e30

    earth_mass = 5.972e24
    earth_speed = 3.029e4
    earth_sun_distance = 147.1e9
    earth_sun_angle = 20 * math.pi/180

    moon_mass = 0.07346e24
    moon_speed = 1.1e3
    moon_earth_distance = 3.57e8
    moon_earth_angle = 0 * math.pi/180

    sun = Body(mass=sun_mass,
               position=[0, 0],
               velocity=[0, 0],
               color='yellow')

    earth_sun_position = polar(earth_sun_distance, earth_sun_angle)
    earth_sun_velocity = polar(earth_speed, earth_sun_angle, 1)

    earth = Body(mass=earth_mass,
                position=earth_sun_position,
                velocity=earth_sun_velocity,
                color='green')

    moon_earth_position = polar(moon_earth_distance, moon_earth_angle)
    moon_earth_velocity = polar(moon_speed, moon_earth_angle, 1)
    
    moon = Body(mass=moon_mass,
                position=earth_sun_position + moon_earth_position,
                velocity=earth_sun_velocity + moon_earth_velocity,
                color='gray')

    bodies = [sun, earth]
    reference_body = 0      #The body you'll be having at the center; the number is the index in the array

    mass = np.array([item.mass for item in bodies]).astype('float64')  
    pos = np.array([item.position for item in bodies]).astype('float64')
    vel = np.array([item.velocity for item in bodies]).astype('float64')
    colors = [item.color for item in bodies]

    N = mass.shape[0]

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE, PE = getEnergy(pos, vel, mass, G)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos
    KE_save = np.zeros(Nt + 1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt + 1)
    PE_save[0] = PE
    t_all = np.arange(Nt + 1) * dt

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        
        vel += acc * dt / 2.0
        
        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE, PE = getEnergy(pos, vel, mass, G)

        bias = -pos[reference_body]

        # save energies, positions for plotting trail
        pos_save[:, 0, i + 1] = pos[:, 0] + len(pos)*[bias[0]]
        pos_save[:, 1, i + 1] = pos[:, 1] + len(pos)*[bias[1]]
        KE_save[i + 1] = KE
        PE_save[i + 1] = PE

        scale = [max(math.log(m[0], log_base), min_size) for m in mass]

    


        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:, 0, max(i - trail_len, 0):i + 1]
            yy = pos_save[:, 1, max(i - trail_len, 0):i + 1]
            plt.scatter(xx, yy, s=1, color=trail_color)
            plt.scatter(pos[:, 0] + len(pos)*[bias[0]], pos[:, 1] + len(pos)*[bias[1]], s=scale, color=colors)
            ax1.set_facecolor('black')
            max_pos = max(pos[0])
            ax1.set(xlim=(looking_range[0][0], looking_range[0][1]), ylim=(looking_range[1][0], looking_range[1][1]))
            ax1.set_aspect('equal', 'box')
            #ax1.set_xticks([-2, -1, 0, 1, 2])
            #ax1.set_yticks([-2, -1, 0, 1, 2])

            plt.sca(ax2)
            plt.cla()
            plt.scatter(t_all, KE_save, color='red', s=1, label='KE' if i == Nt - 1 else "")
            plt.scatter(t_all, PE_save, color='blue', s=1, label='PE' if i == Nt - 1 else "")
            plt.scatter(t_all, KE_save + PE_save, color='black', s=1, label='Etot' if i == Nt - 1 else "")
            ax2.set(xlim=(0, tEnd), ylim=(-500, 500))
            ax2.set_aspect(0.007)
            ax1.set(xlabel="10^8")
            ax1.set(ylabel="10^8")

            plt.pause(0.00001)

    # add labels/legend
    #plt.sca(ax2)
    plt.xlabel('time')
    plt.ylabel('energy')
    ax2.legend(loc='upper right')

    # Save figure
    plt.savefig('nbody.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
