import diffrax
import jax
import jax.numpy as jnp
from jax import lax
from jax import jit
from jax.scipy.sparse.linalg import gmres
from jax.scipy.special import gammaln, logsumexp
from functools import partial


class AdaptiveMidpoint(diffrax.AbstractSolver):
    rtol: float = 1e-6
    atol: float = 1e-8
    max_iters: int = 300

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 2

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        Nx, Ny, Nz, Nn, Nm, Np, Ns = args[:7]
        alpha_tol, u_tol = args[-2:]

        Ck_Fk0, alpha_s, u_s = y0

        δt = t1 - t0
        f0 = terms.vf(t0, y0, args)[0]
        Ck_Fk1_init = Ck_Fk0 + δt * f0


        # Define F(y1) = y1 - y0 - δt * f(t1, (y0 + y1)/2)
        def F_fn(Ck_Fk1):
            Ck_Fk_mid = 0.5 * (Ck_Fk0 + Ck_Fk1)
            return Ck_Fk1 - Ck_Fk0 - δt * terms.vf(t1, (Ck_Fk_mid, alpha_s, u_s), args)[0]


        Ck_Fk1 = _newton_gmres(F_fn, Ck_Fk0, Ck_Fk1_init, self.rtol, self.atol, self.max_iters)

        Ck_Fk_error = Ck_Fk1 - Ck_Fk1_init

        total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz
        Ck1 = Ck_Fk1[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz)
        Ck1 = Ck1.reshape(Ns, Np, Nm, Nn, *Ck1.shape[-3:])

        condition, alpha_new, u_new = basis_change_conditions(Ck1, Nx, Ny, Nz, Ns, alpha_s, u_s, alpha_tol, u_tol)
        Ck1, alpha_s, u_s = lax.cond(condition,
                                         lambda _: change_hermite_basis(Ck1,
                                                                        Nn, Nm,
                                                                        Np, Ns,
                                                                        Nx, Ny,
                                                                        Nz,
                                                                        alpha_s,
                                                                        u_s,
                                                                        alpha_new,
                                                                        u_new),
                                         lambda _: (Ck1, alpha_s, u_s),
                                         None)

        Ck1 = Ck1.flatten()
        y1 = (jnp.concatenate([Ck1, Ck_Fk1[total_Ck_size:]]), alpha_s, u_s)

        dense_info = dict(y0=y0, y1=y1)
        return y1, Ck_Fk_error, dense_info, None, diffrax.RESULTS.successful


def _newton_gmres(F_fn, y0, y_init, rtol, atol, max_iters):
    """Optimized Newton-GMRES with adaptive tolerances and single F evaluation."""
    
    @jax.jit
    def loop_fn(y_init):
        def cond_fn(state):
            _, not_converged, i = state
            return (i < max_iters) & not_converged

        def body_fn(state):
            y1, _, i = state
            
            res, jvp = jax.linearize(F_fn, y1)
            
            scale = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
            norm = jnp.linalg.norm(res / scale)
            
            # Adaptive inner tolerance (Eisenstat-Walker)
            inner_tol = jnp.minimum(0.1, norm * 0.5)
            
            delta, _ = gmres(jvp, -res, 
                           tol=inner_tol, 
                           atol=atol, 
                           maxiter=min(20, max_iters//2))

            # delta = precond_fn(delta_tilde)  # Apply preconditioner to the solution

            y1_next = jnp.where(norm < 1.0, y1, y1 + delta)
            
            return y1_next, norm >= 1.0, i + 1

        init_state = (y_init, True, 0)
        y1_final, _, _ = lax.while_loop(cond_fn, body_fn, init_state)
        return y1_final

    return loop_fn(y_init)

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Ns'])
def basis_change_conditions(Ck, Nx, Ny, Nz, Ns, alpha_s, u_s, alpha_tol, u_tol):
    C_avg = Ck[:, :, :, :, (Ny - 1) // 2, (Nx - 1) // 2, (Nz - 1) // 2].real

    alpha = alpha_s.reshape(Ns, 3)
    alpha_new_x = alpha[:, 0] * jnp.sqrt(1 + jnp.sqrt(2) * C_avg[:, 0, 0, 2] / C_avg[:, 0, 0, 0] - (C_avg[:, 0, 0, 1] / C_avg[:, 0, 0, 0]) ** 2)
    alpha_new_y = alpha[:, 1] #* jnp.sqrt(1 + jnp.sqrt(2) * C_avg[:, 0, 2, 0] / C_avg[:, 0, 0, 0] - (C_avg[:, 0, 1, 0] / C_avg[:, 0, 0, 0]) ** 2)
    alpha_new_z = alpha[:, 2] #* jnp.sqrt(1 + jnp.sqrt(2) * C_avg[:, 2, 0, 0] / C_avg[:, 0, 0, 0] - (C_avg[:, 1, 0, 0] / C_avg[:, 0, 0, 0]) ** 2)
    alpha_new = jnp.stack([alpha_new_x, alpha_new_y, alpha_new_z], axis=1)
    alpha_new = alpha_new.flatten()
    alpha_new = alpha_new.real

    u = u_s.reshape(Ns, 3)
    u_new_x = u[:, 0] + alpha[:, 0] / jnp.sqrt(2) * C_avg[:, 0, 0, 1] / C_avg[:, 0, 0, 0]
    u_new_y = u[:, 1] #+ alpha[:, 1] / jnp.sqrt(2) * C_avg[:, 0, 1, 0] / C_avg[:, 0, 0, 0]
    u_new_z = u[:, 2] #+ alpha[:, 2] / jnp.sqrt(2) * C_avg[:, 1, 0, 0] / C_avg[:, 0, 0, 0]
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
        log_K = (m - n) / 2 * jnp.log(2) - ln_factorial(m) / 2 + ln_factorial(
            n) / 2
        all_sum_indices = jnp.arange(0, num_hermite[dim] + 1)
        mask = (all_sum_indices >= m) & (all_sum_indices <= n) & (
                    (all_sum_indices - m) % 2 == 0)

        base1 = jnp.complex128(-2.0 * b / a)
        exp1 = n - all_sum_indices
        log_base1 = jnp.log(jnp.where(base1 == 0, 1.0, base1))
        term1 = jnp.where(base1 == 0,
                          jnp.where(exp1 == 0, 0.0, -jnp.inf),
                          exp1 * log_base1)

        base2 = jnp.complex128(1.0 / a ** 2 - 1.0)
        exp2 = (all_sum_indices - m) / 2.0
        log_base2 = jnp.log(jnp.where(base2 == 0, 1.0, base2))
        term2 = jnp.where(base2 == 0,
                          jnp.where(exp2 == 0, 0.0, -jnp.inf),
                          exp2 * log_base2)

        log_summand = (-ln_factorial(n - all_sum_indices) - ln_factorial(
            (all_sum_indices - m) / 2.0)
                       + term1 + term2)

        log_summand_masked = jnp.where(mask, log_summand, -jnp.inf)
        to_sum = log_K - (m + 1) * jnp.log(a) + log_summand_masked
        log_sum = logsumexp(to_sum)

        return lax.select(n >= m, jnp.exp(log_sum), jnp.complex128(0))

    def transform_species(species, Ck):
        Ck_species = Ck[species]

        Px = jax.vmap(jax.vmap(transformation_matrix, in_axes=(None, 0, None, None)),
                    in_axes=(0, None, None, None))(jnp.arange(Nn),
                                             jnp.arange(Nn), species, 0)

        Py = jax.vmap(jax.vmap(transformation_matrix, in_axes=(None, 0, None, None)),
                    in_axes=(0, None, None, None))(jnp.arange(Nm),
                                             jnp.arange(Nm), species, 1)

        Pz = jax.vmap(jax.vmap(transformation_matrix, in_axes=(None, 0, None, None)),
                    in_axes=(0, None, None, None))(jnp.arange(Np),
                                            jnp.arange(Np), species, 2)

        Ck_species = jnp.einsum('ip,jq,kr,rqpyxz->kjiyxz', Px, Py, Pz, Ck_species)

        Ck = Ck.at[species].set(Ck_species)

        return Ck

    Ck_new = lax.fori_loop(0, Ns, transform_species, Ck)
    return Ck_new, alpha_new, u_new