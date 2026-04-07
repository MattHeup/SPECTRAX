import os
import sys
import time
import queue
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spectrax import simulation, load_parameters


def setup_two_stream(ip, sp):
    import jax.numpy as jnp
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


def sim_worker(toml_path, setup_func_name, sp_kwargs, num_trials, q):
    """
    Isolated worker process to run a single simulation and compute f(x,v).
    Needed to allow timeout of stalled simulations.
    """
    import jax
    import jax.numpy as jnp
    import __main__
    from spectrax._inverse_transform import inverse_HF_transform
    
    try:
        base_ip, base_sp = load_parameters(toml_path)
        
        curr_sp = base_sp.copy()
        curr_sp.update(sp_kwargs)
            
        setup_func = getattr(__main__, setup_func_name)
        curr_ip = setup_func(base_ip, curr_sp)

        warmup_ip = curr_ip.copy()
        warmup_ip["t_max"] = 1e-5
        warmup_sp = curr_sp.copy()
        warmup_sp["timesteps"] = 2
        jax.block_until_ready(simulation(warmup_ip, **warmup_sp))

        t_start = time.perf_counter()
        for i in range(num_trials):
            out = simulation(curr_ip, **curr_sp)
            jax.block_until_ready(out["Ck"])
        t_end = time.perf_counter()

        Ck_final = out["Ck"][-1:]
        u_final = out["u_s"][-1:]
        alpha_final = out["alpha_s"][-1:]
        Nn = curr_sp["Nn"]

        vx = jnp.linspace(-4.0, 4.0, 401)
        Vx, Vy, Vz = jnp.meshgrid(vx, jnp.array([0.]), jnp.array([0.]), indexing='xy')
        
        xi_x = (Vx[None] - u_final[:, 0, None, None, None]) / alpha_final[:, 0, None, None, None]
        xi_y = (Vy[None] - u_final[:, 1, None, None, None]) / alpha_final[:, 1, None, None, None]
        xi_z = (Vz[None] - u_final[:, 2, None, None, None]) / alpha_final[:, 2, None, None, None]
        
        f1 = inverse_HF_transform(Ck_final[:, :Nn, ...], Nn, 1, 1, xi_x, xi_y, xi_z)
        f1_final = np.array(f1[0, 0, :, 0, 0, :, 0])

        res = {
            "exec_time": (t_end - t_start) / num_trials,
            "EM_energy": np.array(out["EM_energy"]),
            "f1_final": f1_final,
            "N": Nn
        }
            
        q.put(("success", res))
        
    except Exception as e:
        import traceback
        q.put(("error", traceback.format_exc()))

def run_with_timeout(toml_name, setup_func_name, sp_kwargs, num_trials, timeout):
    toml_path = os.path.join(os.path.dirname(__file__), toml_name)
    
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=sim_worker, args=(toml_path, setup_func_name, sp_kwargs, num_trials, q))
    p.start()
    
    try:
        status, result = q.get(timeout=timeout)
        p.join()
        if status == "success":
            return result
        else:
            print(f"  -> Simulation failed with error:\n{result}")
            return None
    except queue.Empty:
        print(f"  -> TIMEOUT reached ({timeout}s). Terminating stalled run.")
        p.terminate()
        p.join()
        return None


def run_two_stream_benchmark(N_list, truth_N, num_trials=3, timeout=300):
    toml_name = "input_1D_two_stream.toml"
    
    print(f"--- Generating High-N Static Ground Truth (N={truth_N}) ---")
    truth_static = run_with_timeout(toml_name, "setup_two_stream", {"Nn": truth_N, "adaptive_basis": False}, 1, timeout=timeout*10)
    
    print(f"--- Generating High-N Adaptive Ground Truth (N={truth_N}) ---")
    truth_adaptive = run_with_timeout(toml_name, "setup_two_stream", {"Nn": truth_N, "adaptive_basis": True}, 1, timeout=timeout*10)
    
    if not truth_static or not truth_adaptive:
        print("CRITICAL: One or both ground truth generations failed!")
        return None

    results = {"static": {"time": [], "error_EM": [], "error_f1": [], "N": []},
               "adaptive": {"time": [], "error_EM": [], "error_f1": [], "N": []}}

    for N in N_list:
        for adaptive in [False, True]:
            mode = "adaptive" if adaptive else "static"
            print(f"  Testing {mode.upper()} basis with N={N}...")
            
            res = run_with_timeout(toml_name, "setup_two_stream", {"Nn": N, "adaptive_basis": adaptive}, num_trials, timeout)
            
            if res is None:
                continue

            truth_EM = truth_adaptive["EM_energy"] if adaptive else truth_static["EM_energy"]
            truth_f1 = truth_adaptive["f1_final"] if adaptive else truth_static["f1_final"]

            err_EM = float(np.sum((res["EM_energy"] - truth_EM)**2) / np.sum(truth_EM**2))
            err_f1 = float(np.sum((res["f1_final"] - truth_f1)**2) / np.sum(truth_f1**2))

            results[mode]["time"].append(res["exec_time"])
            results[mode]["error_EM"].append(err_EM)
            results[mode]["error_f1"].append(err_f1)
            results[mode]["N"].append(N)
                
    return results

if __name__ == "__main__":
    mp.freeze_support()
    
    N_test_list = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150] 
    TRUTH_N = 200
    NUM_TRIALS = 3
    TIMEOUT_SECONDS = 60
    
    res = run_two_stream_benchmark(N_test_list, TRUTH_N, NUM_TRIALS, TIMEOUT_SECONDS)

    if not res:
        sys.exit("Benchmarking aborted due to ground truth failure.")

    speedups = []
    speedup_N = []
    for n in N_test_list:
        if n in res["static"]["N"] and n in res["adaptive"]["N"]:
            idx_s = res["static"]["N"].index(n)
            idx_a = res["adaptive"]["N"].index(n)
            time_static = res["static"]["time"][idx_s]
            time_adaptive = res["adaptive"]["time"][idx_a]
            speedups.append(time_static / time_adaptive)
            speedup_N.append(n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax0 = axes[0]
    if res["static"]["N"]:
        ax0.loglog(res["static"]["time"], res["static"]["error_EM"], marker='o', linestyle='-', label='Static', color='tab:blue')
        for i, N in enumerate(res["static"]["N"]):
            ax0.annotate(f"N={N}", (res["static"]["time"][i], res["static"]["error_EM"][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='tab:blue')
    if res["adaptive"]["N"]:
        ax0.loglog(res["adaptive"]["time"], res["adaptive"]["error_EM"], marker='s', linestyle='--', label='Adaptive', color='tab:orange')
        for i, N in enumerate(res["adaptive"]["N"]):
            ax0.annotate(f"N={N}", (res["adaptive"]["time"][i], res["adaptive"]["error_EM"][i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='tab:orange')
    ax0.set_title("EM Energy Error", fontweight='bold')
    ax0.set_xlabel("Execution Time (s)")
    ax0.set_ylabel(r"Relative $L_2$ Error")
    ax0.grid(True, which="both", ls="--", alpha=0.5)
    ax0.legend()

    ax1 = axes[1]
    if res["static"]["N"]:
        ax1.loglog(res["static"]["time"], res["static"]["error_f1"], marker='o', linestyle='-', label='Static', color='tab:blue')
        for i, N in enumerate(res["static"]["N"]):
            ax1.annotate(f"N={N}", (res["static"]["time"][i], res["static"]["error_f1"][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='tab:blue')
    if res["adaptive"]["N"]:
        ax1.loglog(res["adaptive"]["time"], res["adaptive"]["error_f1"], marker='s', linestyle='--', label='Adaptive', color='tab:orange')
        for i, N in enumerate(res["adaptive"]["N"]):
            ax1.annotate(f"N={N}", (res["adaptive"]["time"][i], res["adaptive"]["error_f1"][i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='tab:orange')
    ax1.set_title("Distribution Function $f_e(x,v)$ Error (Final Timestep)", fontweight='bold')
    ax1.set_xlabel("Execution Time (s)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2 = axes[2]
    if speedups:
        ax2.plot(speedup_N, speedups, marker='D', linestyle='-', color='tab:red', linewidth=2)
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.fill_between(speedup_N, 1.0, speedups, where=(np.array(speedups) > 1.0), interpolate=True, color='tab:green', alpha=0.2, label='Adaptive is Faster')
        ax2.fill_between(speedup_N, 1.0, speedups, where=(np.array(speedups) <= 1.0), interpolate=True, color='tab:red', alpha=0.2, label='Static is Faster')
    ax2.set_title("Execution Time Ratio (Static / Adaptive)", fontweight='bold')
    ax2.set_xlabel("Number of Hermite Modes (N)")
    ax2.set_ylabel("Speedup Factor")
    ax2.grid(True, alpha=0.5)
    ax2.legend()

    plt.suptitle("1D Two-Stream Instability Benchmark: Self-Convergence", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_file = "two_stream_final_benchmark.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nBenchmarking complete! Plot saved to {out_file}")
    plt.show()