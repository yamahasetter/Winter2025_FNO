import numpy as np
import dedalus.public as d3

import os 
print('cwd: ', os.getcwd())

# import lo-res initial condition
lo_res_IC = np.load('../../data/run4/0_filtered.npy')[0]

# Parameters
Lx, Lz = 2, 2
Nx, Nz = 32, 32
dealias = 3/2
stop_sim_time = 1e-5
timestepper = d3.RK222
timestep = 1e-8
dtype = np.float64

def solve_from_IC(IC):
    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
    zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

    # Fields
    u = dist.Field(name='u', bases=(xbasis,zbasis))

    # Substitutions
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)

    # Problem
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) + lap(u+lap(u))=0")

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions is user defined
    u['g'] = IC

    # Main loop
    u.change_scales(1)
    u_list = [np.copy(u['g'])]
    # t_list = [solver.sim_time]
    while solver.proceed:
        solver.step(timestep)
        u.change_scales(1)
        u_list.append(np.copy(u['g']))

    return np.array(u_list)


# push forward in time
out = solve_from_IC(lo_res_IC)

# save results
np.save('lo_res_dedalus_solution.npy', out)