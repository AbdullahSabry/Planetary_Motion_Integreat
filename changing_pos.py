import numpy as np
import math
import matplotlib.pyplot as plt

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

m_r = 1.0e24 #mass rescaler
r_r = 1.0e8  #length rescaler
def getAcc(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    reference_particles = 10 # All the major solar system bodies (Planets & the moon)
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

    # The :reference_particles is used to limit the particles used to calculate the acceleration.
    # When using 10 reference particles we only consider the acceleration of all the bodies with respect to these bodies only
    # This is used such that the asteriods generated don't interact with each other, but only interact with the planets and the sun alone

    ax = G * (dx * inv_r3)[:, :reference_particles] @ mass[:reference_particles]
    ay = G * (dy * inv_r3)[:, :reference_particles] @ mass[:reference_particles]
    az = G * (dz * inv_r3)[:, :reference_particles] @ mass[:reference_particles]

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

    # using the same mechanism to only consider the energies of the planets
    reference_particles = 10
    pos = pos[:reference_particles]
    vel = vel[:reference_particles]
    mass = mass[:reference_particles]

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

        # Putting in the mass and rescaling it accordingly
        self.mass = np.array([mass/m_r])
        self.position = np.array([position[0], position[1], 0])/r_r
        self.velocity = np.array([velocity[0], velocity[1], 0])/v_r
        self.color = color

def polar(r, theta, normal=0):

    # When normal is true, it generates a normal vector to theta (equivelant to theta + 90 )
    if normal:
        return np.array([-r * math.sin(theta), r * math.cos(theta)])
    return np.array([r * math.cos(theta), r * math.sin(theta)])

def create_bodies():

    # Creating the solar system bodies

    # Create all the parameters to create all the solar system planets

    sun_mass = 1.9891e30

    earth_mass = 5.972e24
    earth_speed = 3.029e4
    earth_sun_distance = 147.1e9
    earth_sun_angle = 0 * math.pi/180

    moon_mass = 0.07346e24
    moon_speed = 1.1e3
    moon_earth_distance = 3.57e8
    moon_earth_angle = 0 * math.pi/180

    mercury_mass = 0.33010e24
    mercury_speed = 38.86e3
    mercury_sun_distance = 69.818e9
    mercury_sun_angle = 0 * math.pi/180

    venus_mass = 4.8673e24
    venus_speed = 34.79e3
    venus_sun_distance = 108.941e9
    venus_sun_angle = 0 * math.pi/180

    mars_mass = 0.64169e24
    mars_speed = 21.97e3
    mars_sun_distance = 249.261e9
    mars_sun_angle = 0 * math.pi/180

    jupiter_mass = 1898.13e24
    jupiter_speed = 12.44e3
    jupiter_sun_distance = 816.363e9
    jupiter_sun_angle = 0 * math.pi/180

    saturn_mass = 568.32e24
    saturn_speed = 9.09e3
    saturn_sun_distance = 1506.527e9
    saturn_sun_angle = 0 * math.pi/180

    uranus_mass = 86.811e24
    uranus_speed = 6.49e3
    uranus_sun_distance = 3001.390e9
    uranus_sun_angle = 0 * math.pi/180

    neptune_mass = 102.409e24
    neptune_speed = 5.37e3
    neptune_sun_distance = 4558.857e9
    neptune_sun_angle = 0 * math.pi/180


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

    mercury_sun_position = polar(mercury_sun_distance, mercury_sun_angle)
    mercury_sun_velocity = polar(mercury_speed, mercury_sun_angle, 1)
    mercury = Body(mass=mercury_mass,
                    position=mercury_sun_position,
                    velocity=mercury_sun_velocity,
                    color='#708090')

    venus_sun_position = polar(venus_sun_distance, venus_sun_angle)
    venus_sun_velocity = polar(venus_speed, venus_sun_angle, 1)
    venus = Body(mass=venus_mass,
                    position=venus_sun_position,
                    velocity=venus_sun_velocity,
                    color='#F2F0DF')

    mars_sun_position = polar(mars_sun_distance, mars_sun_angle)
    mars_sun_velocity = polar(mars_speed,mars_sun_angle, 1)
    mars = Body(mass=mars_mass,
                    position=mars_sun_position,
                    velocity=mars_sun_velocity,
                    color='#c1440e')

    jupiter_sun_position = polar(jupiter_sun_distance, jupiter_sun_angle)
    jupiter_sun_velocity = polar(jupiter_speed,jupiter_sun_angle, 1)
    jupiter = Body(mass=jupiter_mass,
                    position=jupiter_sun_position,
                    velocity=jupiter_sun_velocity,
                    color='#c99039')

    saturn_sun_position = polar(saturn_sun_distance, saturn_sun_angle)
    saturn_sun_velocity = polar(saturn_speed, saturn_sun_angle, 1)
    saturn = Body(mass=saturn_mass,
                    position=saturn_sun_position,
                    velocity=saturn_sun_velocity,
                    color='#e3e0c0')
    
    uranus_sun_position = polar(uranus_sun_distance, uranus_sun_angle)
    uranus_sun_velocity = polar(uranus_speed, uranus_sun_angle, 1)
    uranus = Body(mass=uranus_mass,
                    position=uranus_sun_position,
                    velocity=uranus_sun_velocity,
                    color='#c6d3e3')

    neptune_sun_position = polar(neptune_sun_distance, neptune_sun_angle)
    neptune_sun_velocity = polar(neptune_speed, neptune_sun_angle, 1)
    neptune = Body(mass=neptune_mass,
                    position=neptune_sun_position,
                    velocity=neptune_sun_velocity,
                    color='#274687')

    bodies = [sun, mercury, earth, moon, venus, mars, jupiter, saturn, uranus, neptune]
    reference_body = earth   #The body you'll be having at the center; the number is the index in the array

    # Creating Asteriod bodies

    # Generates a cube of asteriods for testing.
    grid_size = 30  # number of samples is grid_size ** 2
    asteriod_speed = 50e3    # Speed of the asteriods
    asteriod_mass = 0.0001e24
    closest_distance = moon_earth_distance       # Closest distance from the cube to the Earth

    """
                . (earth)

        // Body in the middle of the first row is the reference body
          . . . . . . .
          . . . . . . .
          . . . . . . .
          . . . . . . .
          . . . . . . .
          . . . . . . .  //Velocity is up if asteriod_initial_speed_angle = 0. to the left if asteriod_initial_speed_angle = 90 * math.pi/180
          . . . . . . .  // Grid size is the number of rows = columns (here is 7)
    """

    distance_variation = moon_earth_distance/20      # Distance between generated asteriods
    asteriod_initial_speed_angle = 20 * math.pi/180  # Initial angle of asteriod velocity. 0 := directed to earth directly. 90 := tangential (in radians)
    
    
    ref_asteriod_velocity = earth_sun_velocity + polar(asteriod_speed, earth_sun_angle + asteriod_initial_speed_angle, 1) # The velocity with respect to the sun (origin)
    ref_asteroid_distance = earth_sun_position - polar(closest_distance, earth_sun_angle, 1) # Position with respect to the sun (origin)

    grid_size = int(grid_size/2)

    # A for loop to generate the asteriods and append them to the 'bodies' list
    # We use the reference body (directly behind the earth) to calculate the position of all the other bodies
    # By adding to it multiples of the distance variation (distance between two consecutive asteriods)
    # The velocity of all the asteriods is the same as the reference one
    for j in range(2*grid_size + 1):
        for i in range(-grid_size, grid_size+1):
            temp_body = Body(mass=asteriod_mass,
                             position=ref_asteroid_distance+polar(i*distance_variation, earth_sun_angle, 0)-polar(j*distance_variation, earth_sun_angle, 1),
                             velocity=ref_asteriod_velocity,
                             color='blue'   
                                )
            bodies.append(temp_body)


    #returns the bodies / reference body number (the one tracked which is Earth)/ danger_zone limits to be used in main 
    danger_grid_limits = np.array([[-(grid_size + 1) * distance_variation, (grid_size + 1) * distance_variation], [-closest_distance - (2 * grid_size + 1) * distance_variation, -closest_distance + distance_variation]])/r_r
            
    return bodies, bodies.index(reference_body), danger_grid_limits

def main():
    """ N-body simulation """
    # Energy Scaling Constant
    Energy_scaler = 6.67408e29
    time_scaler = 1.41674129703 #scaling of time in days
    m_r = 1.0e24 #mass rescaler
    r_r = 1.0e8  #length rescaler
    # Simulation parameters
    t = 0  # current time of the simulation
    tEnd = 2/time_scaler  # time at which simulation ends
    dt = 0.005  # timestep
    softening = 0  # softening length
    G = 1  # Newton's Gravitational Constant
    plotRealTime = False  # For plotting in real time

    # Visual Parameters
    log_base = 1.3 # Used to change the size of objects (range is from 1 to 2 ideally)
    min_size = 9 #min size of each particle
    square_range = 12 #The range to view in the rescaled units (about 4 is the distance to the moon from the earth)
    looking_range = [[-square_range, square_range], [-square_range, square_range]]      #The range you'll be looking at away from the reference [[xmin, xmax], [ymin, ymax]]
    trail_len = 0
    trail_color = 'gray'
    energy_limit = 10**5.9  # The energy plotting limit (This number is ideal for plotting the energy of the entire solar system)
    show_danger_zone = True # If you want to see the danger zone set as True


    bodies, reference_body, danger_grid_limits = create_bodies()        # Creating the bodies with the function defined above
    safe_distance_squared = (2 * 6.371e6/r_r) ** 2  # The safe distance squared to the Earth (it's set as twice the radius of the Earth)
    

    mass = np.array([item.mass for item in bodies]).astype('float64')  
    pos = np.array([item.position for item in bodies]).astype('float64')
    vel = np.array([item.velocity for item in bodies]).astype('float64')
    colors = np.array([item.color for item in bodies])

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
    fig = plt.figure(figsize=(10, 10), dpi=80)
    grid = plt.GridSpec(4, 4, wspace=0.0, hspace=0.3)

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    if show_danger_zone:
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

    
    # Simulation Main Loop
    scale = [max(math.log(m[0], log_base), min_size) for m in mass] # Scale for particle size in diagram
    danger_particles = np.array([]) # An array to save the indecies of the particles which are dangerous after the simulation is over
    for i in range(Nt):
        bias = -pos[reference_body]
        if i == 0:
            referenced_initial_pos = pos + np.array(len(pos) * [bias])
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

        

        # save energies, positions for plotting trail
        pos_save[:, 0, i + 1] = pos[:, 0] + len(pos)*[bias[0]]
        pos_save[:, 1, i + 1] = pos[:, 1] + len(pos)*[bias[1]]
        KE_save[i + 1] = KE
        PE_save[i + 1] = PE

        referenced_pos = pos + np.array(len(pos) * [bias])  # The position of all the particles with respect to the Earth (reference body to view)
        referenced_distance = np.sum(referenced_pos * referenced_pos, axis=1) # The referenced position with respect to the earth squared (x^2 + y^2 + z^2)
        danger_particles = np.unique(np.append(danger_particles, np.where(referenced_distance < safe_distance_squared)[0])) # The particles where the distance is less than the safe distance
        for particle in danger_particles:
            pos[int(particle)] = np.array([0, 0, 0])
            vel[int(particle)] = np.array([0, 0, 0])


        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:, 0, max(i - trail_len, 0):i + 1]
            yy = pos_save[:, 1, max(i - trail_len, 0):i + 1]
            plt.scatter(xx, yy, s=1, color=trail_color)
            plt.scatter(referenced_pos[:, 0], referenced_pos[:, 1], s=scale, color=colors)
            ax1.set_facecolor('black')
            ax1.set(xlim=(looking_range[0][0], looking_range[0][1]), ylim=(looking_range[1][0], looking_range[1][1]))
            ax1.set_aspect('equal', 'box')
            
            plt.sca(ax2)
            plt.cla()
            ke_plt = plt.scatter(t_all, KE_save, color='red', s=1, label='KE' if i == Nt - 1 else "")
            pe_plt = plt.scatter(t_all, PE_save, color='blue', s=1, label='PE' if i == Nt - 1 else "")
            te_plt = plt.scatter(t_all, KE_save + PE_save, color='black', s=1, label='Etot' if i == Nt - 1 else "")
            plt.xlabel('time')
            plt.ylabel(f'energy in {Energy_scaler} J')
            plt.legend([ke_plt, pe_plt, te_plt], ['KE', 'PE', 'Etot'], loc='upper right')

            ax2.set(xlim=(0, tEnd), ylim=(-energy_limit, energy_limit))
            ax2.set_aspect(0.007)
            ax1.set(xlabel=f"10^8 at time {t*time_scaler:.2f} days")
            ax1.set(ylabel="10^8")

            plt.pause(0.00001)

    if show_danger_zone:

        # Coloring the danger particles in red
        for particle in danger_particles:
            colors[int(particle)] = 'red'
        plt.sca(ax3)
        plt.cla()
        xx = pos_save[:, 0, 0]
        yy = pos_save[:, 1, 0]
        plt.scatter(xx, yy, s=1, color=trail_color)
        plt.scatter(referenced_initial_pos[:, 0], referenced_initial_pos[:, 1], s=scale, color=colors)
        ax3.set_facecolor('black')
        ax3.set(xlim=(danger_grid_limits[0][0], danger_grid_limits[0][1]), ylim=(danger_grid_limits[1][0], danger_grid_limits[1][1]))
        ax3.set_aspect('equal', 'box')
        ax3.set(xlabel="Danger Zones after the running simulation")

        # Two plots (one zoomed out and one zoomed in)
        # Above is zoomed out, below is zoomed in on the particles

        plt.sca(ax4)
        plt.cla()
        plt.scatter(xx, yy, s=1, color=trail_color)
        plt.scatter(referenced_initial_pos[:, 0], referenced_initial_pos[:, 1], s=scale, color=colors)
        ax4.set_facecolor('black')
        ax4.set(xlim=(looking_range[0][0], looking_range[0][1]), ylim=(looking_range[1][0], looking_range[1][1]))
        ax4.set_aspect('equal', 'box')
        ax4.set(xlabel="Danger Zones after the running simulation")


    # add labels/legend
    plt.sca(ax2)
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.legend(loc='upper right')

    # Save figure
    plt.savefig('nbody.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()