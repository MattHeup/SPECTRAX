import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

# Ensure spectrax is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spectrax import simulation, load_parameters
from spectrax._inverse_transform import inverse_HF_transform

# ==============================================================================
# 1. INITIAL CONDITION GENERATOR
# ==============================================================================

def setup_two_stream(ip, sp):
    """Rebuilds the Ck_0 and Fk_0 arrays dynamically."""
    alpha_s = ip["alpha_s"]
    nx, Lx, Omega_cs = ip["nx"], ip["Lx"], ip["Omega_cs"]
    dn1, dn2 = ip["dn1"], ip["dn2"]
    Nx, Ny, Nz = sp["Nx"], sp.get("Ny", 1), sp.get("Nz", 1)
    Nn, Nm, Np = sp["Nn"], sp.get("Nm", 1), sp.get("Np", 1)

    indices = jnp.array([int((Nx-1)/2-nx), int((Nx-1)/2+nx)])
    values  = (dn1 + dn2) * Lx / (4 * jnp.pi * nx * Omega_cs[0])
    Fk_0    = jnp.zeros((6, Ny, Nx, Nz), dtype=jnp.complex128)
    Fk_0    = Fk_0.at[0, int((Ny-1)/2), indices, int((Nz-1)/2)].set(values)

    C10 = jnp.array([
        0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn1,
        1 / (alpha_s[0] ** 3) + 0 * 1j,
        0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn1
    ])
    C20 = jnp.array([
        0 + 1j * (1 / (2 * alpha_s[3] ** 3)) * dn2,
        1 / (alpha_s[3] ** 3) + 0 * 1j,
        0 - 1j * (1 / (2 * alpha_s[3] ** 3)) * dn2
    ])
    idx = jnp.array([int((Nx-1)/2-nx), int((Nx-1)/2), int((Nx-1)/2+nx)])
    Ck_0 = jnp.zeros((2 * Nn * Nm * Np, Ny, Nx, Nz), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[0, int((Ny-1)/2), idx, int((Nz-1)/2)].set(C10)
    Ck_0 = Ck_0.at[Nn * Nm * Np, int((Ny-1)/2), idx, int((Nz-1)/2)].set(C20)

    ip_copy = ip.copy()
    ip_copy["Ck_0"] = Ck_0
    ip_copy["Fk_0"] = Fk_0
    return ip_copy

# ==============================================================================
# 2. EXECUTION & PLOTTING
# ==============================================================================

if __name__ == "__main__":
    toml_path = os.path.join(os.path.dirname(__file__), "input_1D_two_stream.toml")
    base_ip, base_sp = load_parameters(toml_path)

    # RUN STATIC
    print(f"--- Running Static Basis ---")
    sp_static = base_sp.copy()
    sp_static["adaptive_basis"] = False
    ip_static = setup_two_stream(base_ip, sp_static)
    out_static = jax.block_until_ready(simulation(ip_static, **sp_static))

    # RUN ADAPTIVE
    print(f"--- Running Adaptive Basis ---")
    sp_adaptive = base_sp.copy()
    sp_adaptive["adaptive_basis"] = True
    ip_adaptive = setup_two_stream(base_ip, sp_adaptive)
    out_adaptive = jax.block_until_ready(simulation(ip_adaptive, **sp_adaptive))

    print("--- Processing Phase Space Data & Plotting ---")
    
    time_arr = out_static["time"]
    # Select three evenly spaced frames (start, middle, end)
    frames = [0, len(time_arr) // 2, len(time_arr) - 1]
    Lx = base_ip["Lx"]
    
    def get_f_tot(out, frame):
        """Inverse transforms both species and returns the combined electron distribution."""
        Ck = out["Ck"][frame:frame+1]
        u_s = out["u_s"][frame:frame+1]
        alpha_s = out["alpha_s"][frame:frame+1]
        Nn = base_sp["Nn"]
        
        # Determine an appropriate global physical velocity grid
        v_max = 5.0 * jnp.max(out["alpha_s"][0])
        vx = jnp.linspace(-v_max, v_max, 251)
        Vx, Vy, Vz = jnp.meshgrid(vx, jnp.array([0.]), jnp.array([0.]), indexing='xy')
        
        # Population 1
        xi_x1 = (Vx[None] - u_s[:, 0, None, None, None]) / alpha_s[:, 0, None, None, None]
        xi_y1 = (Vy[None] - u_s[:, 1, None, None, None]) / alpha_s[:, 1, None, None, None]
        xi_z1 = (Vz[None] - u_s[:, 2, None, None, None]) / alpha_s[:, 2, None, None, None]
        f1 = inverse_HF_transform(Ck[:, :Nn, ...], Nn, 1, 1, xi_x1, xi_y1, xi_z1)
        
        # Population 2
        xi_x2 = (Vx[None] - u_s[:, 3, None, None, None]) / alpha_s[:, 3, None, None, None]
        xi_y2 = (Vy[None] - u_s[:, 4, None, None, None]) / alpha_s[:, 4, None, None, None]
        xi_z2 = (Vz[None] - u_s[:, 5, None, None, None]) / alpha_s[:, 5, None, None, None]
        f2 = inverse_HF_transform(Ck[:, Nn:2*Nn, ...], Nn, 1, 1, xi_x2, xi_y2, xi_z2)
        
        # Sum both populations to get total f_e(x,v)
        f_tot = f1[0, 0, :, 0, 0, :, 0] + f2[0, 0, :, 0, 0, :, 0]
        return np.array(f_tot), vx

    fig = plt.figure(figsize=(16, 12))
    
    # --- ROW 0 & 1: PHASE SPACE SNAPSHOTS ---
    for i, frame in enumerate(frames):
        t_val = time_arr[frame]
        
        # Static Snapshots
        ax_stat = plt.subplot(3, 3, i + 1)
        f_tot_stat, vx_stat = get_f_tot(out_static, frame)
        
        # Note: f_tot is transposed so axes map nicely to (x, v)
        im_stat = ax_stat.imshow(f_tot_stat.T, extent=(0, Lx, vx_stat[0], vx_stat[-1]),
                                 cmap='jet', origin='lower', interpolation='sinc', aspect='auto')
        plt.colorbar(im_stat, ax=ax_stat)
        ax_stat.set_title(f"Static: $f_e(x,v)$ at $t={t_val:.1f}$")
        ax_stat.set_xlabel("$x / d_e$")
        ax_stat.set_ylabel("$v / c$")

        # Adaptive Snapshots
        ax_adap = plt.subplot(3, 3, i + 4)
        f_tot_adap, vx_adap = get_f_tot(out_adaptive, frame)
        im_adap = ax_adap.imshow(f_tot_adap.T, extent=(0, Lx, vx_adap[0], vx_adap[-1]),
                                 cmap='jet', origin='lower', interpolation='sinc', aspect='auto')
        plt.colorbar(im_adap, ax=ax_adap)
        ax_adap.set_title(f"Adaptive: $f_e(x,v)$ at $t={t_val:.1f}$")
        ax_adap.set_xlabel("$x / d_e$")
        ax_adap.set_ylabel("$v / c$")

    # --- ROW 2, COL 0: DRIFT VELOCITY (u) OVER TIME ---
    ax_u = plt.subplot(3, 3, 7)
    u_adap = out_adaptive["u_s"]
    ax_u.plot(time_arr, u_adap[:, 0], label="Pop 1 ($u_{e1}$)", color="tab:red", lw=2)
    ax_u.plot(time_arr, u_adap[:, 3], label="Pop 2 ($u_{e2}$)", color="tab:green", lw=2)
    ax_u.set_title("Drift Velocity (Adaptive Basis)")
    ax_u.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
    ax_u.set_ylabel("$u / c$")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)

    # --- ROW 2, COL 1: THERMAL SCALE (alpha) OVER TIME ---
    ax_alpha = plt.subplot(3, 3, 8)
    alpha_adap = out_adaptive["alpha_s"]
    ax_alpha.plot(time_arr, alpha_adap[:, 0], label=r"Pop 1 ($\alpha_{e1}$)", color="tab:purple", lw=2)
    ax_alpha.plot(time_arr, alpha_adap[:, 3], label=r"Pop 2 ($\alpha_{e2}$)", color="tab:brown", lw=2)
    ax_alpha.set_title("Thermal Scale (Adaptive Basis)")
    ax_alpha.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
    ax_alpha.set_ylabel(r"$\alpha / c$")
    ax_alpha.legend()
    ax_alpha.grid(True, alpha=0.3)

    # --- ROW 2, COL 2: TOTAL ENERGY ERROR ---
    ax_err = plt.subplot(3, 3, 9)
    # Add a tiny epsilon to avoid log(0)
    err_stat = np.abs(out_static["total_energy"] - out_static["total_energy"][0]) / (out_static["total_energy"][0] + 1e-15)
    err_adap = np.abs(out_adaptive["total_energy"] - out_adaptive["total_energy"][0]) / (out_adaptive["total_energy"][0] + 1e-15)
    
    # Slice from [1:] to skip the t=0 exact zero
    ax_err.plot(time_arr[1:], err_stat[1:], label="Static", color="tab:blue", lw=2)
    ax_err.plot(time_arr[1:], err_adap[1:], label="Adaptive", color="tab:orange", lw=2, linestyle='--')
    ax_err.set_yscale('log')
    ax_err.set_title("Relative Total Energy Error")
    ax_err.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
    ax_err.set_ylabel(r"$|E(t) - E(0)| / E(0)$")
    ax_err.legend()
    ax_err.grid(True, alpha=0.5, which="both", ls="--")

    # Display and Save
    # plt.suptitle(f"1D Two-Stream Instability Dynamics", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_file = r"Examples\tsi_adaptive_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_file}")
    plt.show()