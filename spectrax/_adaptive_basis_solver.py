import diffrax
import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.scipy.special import gammaln, logsumexp
from functools import partial

__all__ = ["AdaptiveBasisWrapper"]


class AdaptiveBasisWrapper(diffrax.AbstractWrappedSolver):
    """
    Wraps any standard Diffrax Runge-Kutta solver to apply an adaptive 
    Hermite-basis transformation when tracking condition thresholds are met.
    """
    solver: diffrax.AbstractRungeKutta
    term_structure = diffrax.ODETerm

    @property
    def interpolation_cls(self):
        return self.solver.interpolation_cls

    def order(self, terms):
        return self.solver.order(terms)

    def init(self, terms, t0, t1, y0, args):
        return self.solver.init(terms, t0, t1, y0, args)

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        y1, error, dense_info, solver_state, result = self.solver.step(terms, t0, t1, y0, args, solver_state, made_jump)

        Nx, Ny, Nz, Nn, Nm, Np, Ns = args[:7]
        alpha_tol, u_tol = args[-2:]
        Ck_Fk1, alpha_s, u_s = y1

        total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz
        Fk1 = Ck_Fk1[total_Ck_size:]
        Ck1 = Ck_Fk1[:total_Ck_size].reshape(Ns, Np, Nm, Nn, Ny, Nx, Nz)

        condition, alpha_new, u_new = basis_change_conditions(Ck1, Nx, Ny, Nz, Ns, alpha_s, u_s, alpha_tol, u_tol)
        
        Ck1_new, alpha_s_new, u_s_new = lax.cond(
            condition,
            lambda _: change_hermite_basis(Ck1, Nn, Nm, Np, Ns, Nx, Ny, Nz, alpha_s, u_s, alpha_new, u_new),
            lambda _: (Ck1, alpha_s, u_s),
            None
        )

        Ck_Fk1_new = jnp.concatenate([Ck1_new.flatten(), Fk1])
        y1_new = (Ck_Fk1_new, alpha_s_new, u_s_new)

        solver_state = lax.cond(
            condition,
            lambda _: self.solver.init(terms, t1, t1 + (t1 - t0), y1_new, args),
            lambda _: solver_state,
            None
        )
        
        dense_info = lax.cond(
            condition,
            lambda _: dict(y0=y0, y1=y1_new, k=dense_info.get('k')),
            lambda _: dense_info,
            None
        )

        return y1_new, error, dense_info, solver_state, result


@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Ns'])
def basis_change_conditions(Ck, Nx, Ny, Nz, Ns, alpha_s, u_s, alpha_tol, u_tol):
    C_avg = Ck[:, :, :, :, (Ny - 1) // 2, (Nx - 1) // 2, (Nz - 1) // 2].real

    alpha = alpha_s.reshape(Ns, 3)
    
    alpha_new_x = alpha[:, 0] * jnp.sqrt(1 + jnp.sqrt(2) * C_avg[:, 0, 0, 2] / C_avg[:, 0, 0, 0] - (C_avg[:, 0, 0, 1] / C_avg[:, 0, 0, 0]) ** 2)
    alpha_new_y = alpha[:, 1] # * jnp.sqrt(1 + jnp.sqrt(2) * C_avg[:, 0, 2, 0] / C_avg[:, 0, 0, 0] - (C_avg[:, 0, 1, 0] / C_avg[:, 0, 0, 0]) ** 2)
    alpha_new_z = alpha[:, 2] # * jnp.sqrt(1 + jnp.sqrt(2) * C_avg[:, 2, 0, 0] / C_avg[:, 0, 0, 0] - (C_avg[:, 1, 0, 0] / C_avg[:, 0, 0, 0]) ** 2)
    
    alpha_new = jnp.stack([alpha_new_x, alpha_new_y, alpha_new_z], axis=1)
    alpha_new = alpha_new.flatten()

    u = u_s.reshape(Ns, 3)
    
    u_new_x = u[:, 0] + alpha[:, 0] / jnp.sqrt(2) * C_avg[:, 0, 0, 1] / C_avg[:, 0, 0, 0]
    u_new_y = u[:, 1] # + alpha[:, 1] / jnp.sqrt(2) * C_avg[:, 0, 1, 0] / C_avg[:, 0, 0, 0]
    u_new_z = u[:, 2] # + alpha[:, 2] / jnp.sqrt(2) * C_avg[:, 1, 0, 0] / C_avg[:, 0, 0, 0]
    
    u_new = jnp.stack([u_new_x, u_new_y, u_new_z], axis=1)
    u_new = u_new.flatten()

    condition_met = jnp.any(jnp.logical_or(jnp.abs(alpha_new - alpha_s) > alpha_tol, jnp.abs(u_new - u_s) > u_tol))

    return condition_met, alpha_new, u_new


@partial(jit, static_argnames=['Nx', 'Ny', 'Nz','Nn', 'Nm', 'Np', 'Ns'])
def change_hermite_basis(Ck, Nn, Nm, Np, Ns, Nx, Ny, Nz, alpha_old, u_old, alpha_new, u_new):
    num_hermite = (Nn, Nm, Np)

    def ln_factorial(n):
        return gammaln(n + 1)

    def transformation_matrix(n, m, species, dim):
        a = alpha_new[species * 3 + dim] / alpha_old[species * 3 + dim]
        b = (u_new[species * 3 + dim] - u_old[species * 3 + dim]) / alpha_old[species * 3 + dim]
        
        log_K = (m - n) / 2 * jnp.log(2) - ln_factorial(m) / 2 + ln_factorial(n) / 2
        
        all_sum_indices = jnp.arange(0, num_hermite[dim] + 1)
        mask = (all_sum_indices >= m) & (all_sum_indices <= n) & ((all_sum_indices - m) % 2 == 0)

        base1 = jnp.complex128(-2.0 * b / a)
        exp1 = n - all_sum_indices
        log_base1 = jnp.log(jnp.where(base1 == 0, 1.0, base1))
        term1 = jnp.where(base1 == 0, jnp.where(exp1 == 0, 0.0, -jnp.inf), exp1 * log_base1)

        base2 = jnp.complex128(1.0 / a ** 2 - 1.0)
        exp2 = (all_sum_indices - m) / 2.0
        log_base2 = jnp.log(jnp.where(base2 == 0, 1.0, base2))
        term2 = jnp.where(base2 == 0, jnp.where(exp2 == 0, 0.0, -jnp.inf), exp2 * log_base2)

        log_summand = -ln_factorial(n - all_sum_indices) - ln_factorial((all_sum_indices - m) / 2.0) + term1 + term2
        log_summand_masked = jnp.where(mask, log_summand, -jnp.inf)
        
        to_sum = log_K - (m + 1) * jnp.log(a) + log_summand_masked
        log_sum = logsumexp(to_sum)

        return lax.select(n >= m, jnp.exp(log_sum), jnp.complex128(0))

    def transform_species(species, Ck_arr):
        Ck_species = Ck_arr[species]

        Px = jax.vmap(jax.vmap(transformation_matrix, in_axes=(None, 0, None, None)),
                    in_axes=(0, None, None, None))(jnp.arange(Nn), jnp.arange(Nn), species, 0)

        Py = jax.vmap(jax.vmap(transformation_matrix, in_axes=(None, 0, None, None)),
                    in_axes=(0, None, None, None))(jnp.arange(Nm), jnp.arange(Nm), species, 1)

        Pz = jax.vmap(jax.vmap(transformation_matrix, in_axes=(None, 0, None, None)),
                    in_axes=(0, None, None, None))(jnp.arange(Np), jnp.arange(Np), species, 2)

        Ck_species = jnp.einsum('ip,jq,kr,rqpyxz->kjiyxz', Px, Py, Pz, Ck_species)
        return Ck_arr.at[species].set(Ck_species)

    Ck_new = lax.fori_loop(0, Ns, transform_species, Ck)
    return Ck_new, alpha_new, u_new