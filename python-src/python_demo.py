import taichi as ti
import numpy as np
import argparse
ti.init(arch=ti.cuda, default_fp = ti.f32, device_memory_GB=16, debug=False, kernel_profiler=True, fast_math=False)


n_grid = 256
n_max_particles = 2000000

dx = 1 / n_grid
rho = 1000
particle_volume =  dx * dx * dx / 8
particle_mass = particle_volume * rho
gravity = -4

# E = 1e5
E = 5e3
nu = 0.4

cfl = 0.5

mu = E / (2 * (1 + nu))
lamb = E * nu / ((1 + nu) * (1 - 2 * nu))


use_ckmpm = True
twopi = 2 * np.pi


mpm_float = ti.float32
grid_domain = (n_grid, n_grid, n_grid, 2) if use_ckmpm else (n_grid, n_grid, n_grid)

axes = ti.ijkl if use_ckmpm else ti.ijk
parent_dimension_prefix = (1, ) if use_ckmpm else ()
dimension_prefix = (2, ) if use_ckmpm else ()

mass_grid = ti.field(dtype=mpm_float)
mass_grid_root = ti.root.pointer(axes, (16, 16, 16) + parent_dimension_prefix)
mass_grid_pointer = mass_grid_root.pointer(axes, (8, 8, 8) + parent_dimension_prefix)
mass_grid_dense = mass_grid_pointer.dense(axes, (4, 4, 4) + dimension_prefix)
mass_grid_dense.place(mass_grid)

momentum_grid = ti.Vector.field(n = 3, dtype= mpm_float)
momentum_grid_root = ti.root.pointer(axes, (16, 16, 16) + parent_dimension_prefix)
momentum_grid_pointer = momentum_grid_root.pointer(axes, (8, 8, 8) + parent_dimension_prefix)
momentum_grid_dense = momentum_grid_pointer.dense(axes, (4, 4, 4) + dimension_prefix)
momentum_grid_dense.place(momentum_grid)

particle_position = ti.Vector.field(n = 3, dtype= mpm_float, shape = (n_max_particles, ))
particle_velocity = ti.Vector.field(n = 3, dtype= mpm_float, shape = (n_max_particles, ))
particle_C = ti.Matrix.field(n = 3, m = 3, dtype= mpm_float, shape = (n_max_particles, ))
particle_F = ti.Matrix.field(n = 3, m = 3, dtype= mpm_float, shape = (n_max_particles, ))


def compute_sound_cfl_dt(cfl: float, dx: float, E: float, nu: float, rho: float):
    return cfl * dx / np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))

default_dt = compute_sound_cfl_dt(cfl, dx, E, nu, rho) * 0.1

@ti.func
def compact_kernel_stencil(dw: ti.template()):
    
    stencil = ti.Matrix.zero(mpm_float, 3, 2)
    for i in ti.static(range(3)):
        stencil[i, 0] = 1 - dw[i] + ti.sin(twopi * dw[i]) / twopi
        stencil[i, 1] = 1 - stencil[i, 0]
    return stencil

@ti.func
def compact_kernel_gradient(cell: ti.template(), stencil: ti.template(), dw: ti.template()):
    kx, ky, kz = stencil[0, cell[0]], stencil[1, cell[1]], stencil[2, cell[2]]
    grad = ti.math.sign(dw) * (ti.cos(twopi * ti.abs(dw)) - 1)
    
    return n_grid * ti.Vector([ky * kz, kx * kz, kx * ky]) * grad

@ti.func
def quadratic_kernel_gradient(cell: ti.template(), stencil: ti.template(), dw: ti.template()):
    kx, ky, kz = stencil[0], stencil[1], stencil[2]
    grad = (0 <= ti.abs(dw) < 0.5) * -2 * dw + (0.5 <= ti.abs(dw) < 1.5) * (ti.abs(dw) - 1.5) * ti.math.sign(dw)

    return n_grid * grad * ti.Vector([ky * kz, kx * kz, kx * ky])


@ti.func
def compute_fixed_corotated(F: ti.template()):

    U, Sigma, V = ti.svd(F)
    R = U @ V.transpose()
    J = Sigma[0, 0] * Sigma[1, 1] * Sigma[2, 2]

    return -particle_volume * (2 * mu * (F - R) @ F.transpose() + lamb * (J - 1) * J * ti.Matrix.identity(mpm_float, 3))

@ti.kernel
def compute_cfl_dt(particle_count: ti.i32) -> ti.f32:
    min_dt = default_dt
    for particle_index in range(particle_count):
        ti.atomic_min(min_dt, 0.5 * dx / particle_velocity[particle_index].norm())
    return min_dt


@ti.kernel
def substep(method: ti.template(), apic_mode: ti.template(), dt: ti.f32, particle_count: ti.i32) -> ti.f32:
    for I in ti.grouped(mass_grid): # Clear the grid
        mass_grid[I] = 0
        momentum_grid[I] = 0

    # P2G
    ti.loop_config(block_dim=128)
    for particle_index in range(particle_count):
        global_position = particle_position[particle_index]
        velocity = particle_velocity[particle_index]
        C = ti.Matrix.zero(mpm_float, 3, 3) 
        PF = compute_fixed_corotated(particle_F[particle_index])

        if ti.static(method == 0): # CK_MPM
            for w in ti.static(range(2)):
                sign = ti.static(-1 if w == 0 else 1)
                cell_index = ti.cast(global_position * n_grid - sign * 0.25, ti.i32)

                offset = global_position * n_grid - (cell_index + 0.25 * sign)
                stencil = compact_kernel_stencil(offset) 
                for I in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                    cell = ti.Vector([cell_index[0] + I[0], cell_index[1] + I[1], cell_index[2] + I[2], w])
                    weight = stencil[0, I[0]] * stencil[1, I[1]] * stencil[2, I[2]]
                    mass_grid[cell] += weight * particle_mass
                    momentum_grid[cell] += weight * particle_mass * velocity + dt * PF @ compact_kernel_gradient(I, stencil, offset - I)



        elif ti.static(method == 1): # Quadratic Kernel
            cell_index = ti.cast(global_position * n_grid - 0.5, ti.i32)
            offset = global_position * n_grid - cell_index

            w = [0.5 * (1.5 - offset) ** 2, 0.75 - (offset - 1.0) ** 2, 0.5 * (offset - 0.5) ** 2]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                I = ti.Vector([i, j, k])
                cell = cell_index + I
                weight = w[i][0] * w[j][1] * w[k][2]
                mass_grid[cell] += weight * particle_mass

                momentum_grid[cell] += weight * particle_mass * velocity + dt * PF @ quadratic_kernel_gradient(I, ti.Vector([w[i][0], w[j][1], w[k][2]]), offset - I)
            
            
    # Grid Update
    max_grid_velocity_norm = 0.0
    ti.loop_config(block_dim=128)
    for I in ti.grouped(mass_grid):
        mass = mass_grid[I]
        velocity = ti.Vector.zero(mpm_float, 3)
        if mass > particle_mass * 1e-8:
            velocity = momentum_grid[I] / mass

        velocity[1] += dt * gravity

        if ti.static(method == 0):
            outofbound = (I[0] < 8) or (I[0] >= 248) or (I[1] < 8) or (I[1] >= 248) or (I[2] < 8) or (I[2] >= 248)
            momentum_grid[I] = 0 if outofbound else velocity

        elif ti.static(method == 1):
            outofbound = (I[0] < 8) or (I[0] >= 248) or (I[1] < 8) or (I[1] >= 248) or (I[2] < 8) or (I[2] >= 248)
            momentum_grid[I] = 0 if outofbound else velocity

        ti.atomic_max(max_grid_velocity_norm, velocity.norm())

    # G2P
    ti.loop_config(block_dim=128)
    for particle_index in range(particle_count):
        global_position = particle_position[particle_index]

        velocity = ti.Vector.zero(mpm_float, 3)
        # B = ti.Matrix.zero(mpm_float, 3, 3)
        # D = ti.Matrix.zero(mpm_float, 3, 3)
        covariantVelocity = ti.Matrix.zero(mpm_float, 3, 3)

        if ti.static(method == 0): # CKMPM
            for w in ti.static(range(2)):
                sign = ti.static(-1 if w == 0 else 1)
                cell_index = ti.cast(global_position * n_grid - sign * 0.25, ti.i32)
                offset = global_position * n_grid - (cell_index + 0.25 * sign)
                stencil = compact_kernel_stencil(offset) 
                for I in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                    cell = ti.Vector([cell_index[0] + I[0], cell_index[1] + I[1], cell_index[2] + I[2], w])
                    weight = stencil[0, I[0]] * stencil[1, I[1]] * stencil[2, I[2]]
                    grid_velocity = momentum_grid[cell]
                    velocity += weight * grid_velocity 

                    covariantVelocity += grid_velocity.outer_product(compact_kernel_gradient(I, stencil, offset - I))
            velocity *= 0.5
            particle_position[particle_index] += dt * velocity
            particle_velocity[particle_index] = velocity
            particle_F[particle_index] = (ti.Matrix.identity(mpm_float, 3) + 0.5 * dt * covariantVelocity) @ particle_F[particle_index]

        elif ti.static(method == 1): # Quadratic Kernel

            cell_index = ti.cast(global_position * n_grid - 0.5, ti.i32)
            offset = global_position * n_grid - cell_index

            w = [0.5 * (1.5 - offset) ** 2, 0.75 - (offset - 1.0) ** 2, 0.5 * (offset - 0.5) ** 2]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                I = ti.Vector([i, j, k])
                cell = cell_index + I
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_velocity = momentum_grid[cell]
                velocity += weight * grid_velocity 

                covariantVelocity += grid_velocity.outer_product(quadratic_kernel_gradient(I, ti.Vector([w[i][0], w[j][1], w[k][2]]), offset - I))

            particle_position[particle_index] += dt * velocity
            particle_velocity[particle_index] = velocity
            particle_F[particle_index] = (ti.Matrix.identity(mpm_float, 3) + dt * covariantVelocity) @ particle_F[particle_index]



    return max_grid_velocity_norm

@ti.kernel
def initialize_particles(particle_count: ti.i32, particle_pos: ti.types.ndarray()):
    for i in range(particle_count):
        particle_position[i][0] = particle_pos[i, 0]
        particle_position[i][1] = particle_pos[i, 1]
        particle_position[i][2] = particle_pos[i, 2]
        particle_F[i] = ti.Matrix.identity(mpm_float, 3)



def main_loop(arguments):

    frame = int(arguments.total_time * arguments.fps + 0.5)

    writer = ti.tools.PLYWriter(num_vertices=arguments.particle_count)
    max_grid_velocity_norm = 0.0
    for i in range(frame):
        time_remain = 1 / arguments.fps

        print(f"Frame {i}")
        substep_idx = 0
        while time_remain > 0:
            mass_grid_root.deactivate_all()
            momentum_grid_root.deactivate_all()

            dt = min(default_dt, compute_cfl_dt(arguments.particle_count))
            dt = min(dt, time_remain)
            if max_grid_velocity_norm > 0.0:
                dt = min(dt, cfl * dx / max_grid_velocity_norm)

            # substep
            max_grid_velocity_norm = substep(0 if use_ckmpm else 1, 1, dt, arguments.particle_count)
            if substep_idx % 100 == 0:
                print(f"\tSubstep: {substep_idx}, Time remain: {time_remain}, Dt: {dt}")
            substep_idx += 1
            time_remain -= dt

        export_pos = particle_position.to_numpy()[:arguments.particle_count]
        writer.add_vertex_pos(export_pos[:, 0], export_pos[:, 1], export_pos[:, 2])
        writer.export_frame(i, arguments.output_dir)



if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process simulation parameters.")

    # Add arguments
    parser.add_argument(
        "--particle_count",
        type=int,
        required=True,
        help="Number of particles in the simulation (integer, required)."
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Output file name (string, required)."
    )
    parser.add_argument(
        "--total_time",
        type=float,
        required=True,
        help="Total simulation time (float, required)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        required=True,
        help="Frames per second for the simulation (integer, required)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Export root directory."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    print(f"Particle Count: {args.particle_count}")
    print(f"Filename: {args.filename}")
    print(f"Total Time: {args.total_time} seconds")
    print(f"FPS: {args.fps}")
    print(f"Default dt: {default_dt}")

    particle_pos = np.fromfile(args.filename, dtype=np.float32).reshape(-1, 3)
    print(particle_pos.shape)
    initialize_particles(args.particle_count, particle_pos)

    main_loop(args)
    ti.profiler.print_kernel_profiler_info()

    



    


        

                    




