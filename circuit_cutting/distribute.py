import math
import os
import shutil
import socket
import subprocess
import time
from collections import defaultdict

import ray
from qiskit.primitives.containers import PrimitiveResult
from qiskit_aer.primitives import SamplerV2

RAY_TASK_CPUS = 1


def log(message):
    print(f"[RAY] {message}", flush=True)


def _available_cpus():
    """CPUs this process may actually use (respects affinity / cgroup-style pinning)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ------------------------------------------------------------
# Remote task (module level so Ray can serialize it cleanly)
# ------------------------------------------------------------
@ray.remote(num_cpus=RAY_TASK_CPUS)
def _run_chunk_remote(circuits, shots):
    """Execute a chunk of circuits on one CPU (Aer restricted to one thread).
    Returns (results, task_stats)."""
    t_start = time.time()
    t0 = time.perf_counter()
    sampler = SamplerV2(options={"backend_options": {"max_parallel_threads": 1,
                                                     "max_parallel_experiments": 1}})
    results = list(sampler.run(circuits, shots=shots).result())
    duration = time.perf_counter() - t0
    stats = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "n_circuits": len(circuits),
        "start_time": t_start,          # wall clock (may differ slightly across nodes)
        "duration": duration,           # measured locally, so reliable
    }
    return results, stats


class RayAgent:
    def __init__(self, num_cpus_per_node=48, port=6379, startup_timeout=180,
                 ray_auth_mode="disabled", local_num_cpus=None):
        self.num_cpus_per_node = num_cpus_per_node
        self.port = port
        self.startup_timeout = startup_timeout
        self.local_num_cpus = local_num_cpus   # CPUs to use in local (non-Slurm) mode; None = all available
        self.nodes = []              # Slurm hostnames
        self.head_node = None
        self.head_ip = None
        self.ray_address = None
        self.total_cpus = 0
        self.ray_nodes = []
        self.init_time = 0.0
        self.cluster_start_time = 0.0
        self.local_mode = False      # True when running a local single-node Ray instance
        self._procs = []
        self._owns_cluster = False
        os.environ.setdefault("RAY_AUTH_MODE", ray_auth_mode)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    @staticmethod
    def slurm_available():
        """True if we are inside a Slurm allocation and Slurm tools are usable."""
        return bool(os.environ.get("SLURM_JOB_NODELIST")) and shutil.which("srun") is not None \
            and shutil.which("scontrol") is not None

    def initialise(self, expected_nodes=None):
        """One-call setup.
        1. If RAY_ADDRESS is set, attach to that existing cluster.
        2. Else if running inside a Slurm allocation, build a cluster over its nodes.
        3. Else (no Slurm, single machine), start a local Ray instance that
           parallelises over the CPUs of this node.
        Then connects the driver."""
        if os.environ.get("RAY_ADDRESS"):
            log(f"RAY_ADDRESS found ({os.environ['RAY_ADDRESS']}); using existing cluster")
            self.ray_address = os.environ["RAY_ADDRESS"]
        elif self.slurm_available():
            self.create_ray_cluster()
        else:
            log("Slurm not available and no RAY_ADDRESS set; using local single-node Ray")
            self.local_mode = True
            expected_nodes = expected_nodes or 1
        return self.connect_ray(expected_nodes=expected_nodes or (len(self.nodes) or None))

    def create_ray_cluster(self):
        """Start head + workers on all nodes of the current Slurm job (replaces the .sh logic)."""
        nodelist = os.environ.get("SLURM_JOB_NODELIST")
        if not nodelist:
            raise RuntimeError("SLURM_JOB_NODELIST not set; run inside a Slurm allocation "
                               "or export RAY_ADDRESS for an existing cluster.")
        t0 = time.perf_counter()
        self.nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True).split()
        self.head_node = self.nodes[0]
        log(f"Allocated nodes ({len(self.nodes)}): {self.nodes}")

        self.get_ip_and_start_head()
        self.start_ray_workers()
        self._owns_cluster = True
        self.cluster_start_time = time.perf_counter() - t0

    def get_ip_and_start_head(self):
        out = subprocess.check_output(
            ["srun", "-N1", "-n1", "-w", self.head_node, "hostname", "-I"], text=True)
        self.head_ip = out.split()[0]
        log(f"Ray head IP: {self.head_ip}")

        proc = subprocess.Popen([
            "srun", "--overlap", "-N1", "-n1", "-w", self.head_node,
            "ray", "start", "--head",
            f"--node-ip-address={self.head_ip}",
            f"--port={self.port}",
            f"--num-cpus={self.num_cpus_per_node}",
            "--block",
        ])
        self._procs.append(proc)
        self.ray_address = f"{self.head_ip}:{self.port}"
        os.environ["RAY_ADDRESS"] = self.ray_address
        log(f"Ray head starting at {self.ray_address}")

    def start_ray_workers(self):
        for node in self.nodes[1:]:
            proc = subprocess.Popen([
                "srun", "--overlap", "-N1", "-n1", "-w", node,
                "ray", "start", f"--address={self.ray_address}",
                f"--num-cpus={self.num_cpus_per_node}",
                "--block",
            ])
            self._procs.append(proc)
        log(f"Launched {len(self.nodes) - 1} Ray workers")

    # --------------------------------------------------------
    # Connect / teardown
    # --------------------------------------------------------
    def _connect_local(self):
        """Start (or reuse) a local single-node Ray instance using this machine's CPUs."""
        num_cpus = self.local_num_cpus or _available_cpus()
        log(f"Starting local Ray instance with {num_cpus} CPUs")
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True,
                 include_dashboard=False, logging_level="WARNING")
        self.ray_address = "local"

    def connect_ray(self, expected_nodes=None):
        """Connect the driver, retrying until the head is up, then wait for workers
        (polls instead of fixed sleeps). In local mode, starts a local Ray instance."""
        t0 = time.perf_counter()

        if self.local_mode:
            self._connect_local()
            address = self.ray_address
        else:
            address = self.ray_address or os.environ.get("RAY_ADDRESS")
            if not address:
                raise RuntimeError("RAY_ADDRESS is not set and no cluster was created.")

            deadline = t0 + self.startup_timeout
            while True:
                try:
                    ray.init(address=address, ignore_reinit_error=True,
                             include_dashboard=False, logging_level="WARNING")
                    break
                except Exception as exc:
                    if time.perf_counter() > deadline:
                        raise RuntimeError(f"Could not connect to Ray at {address}: {exc}")
                    time.sleep(2)

            if expected_nodes:
                while True:
                    alive = [n for n in ray.nodes() if n.get("Alive")]
                    if len(alive) >= expected_nodes:
                        break
                    if time.perf_counter() > deadline:
                        log(f"WARNING: only {len(alive)}/{expected_nodes} nodes joined before timeout")
                        break
                    time.sleep(2)

        self.init_time = time.perf_counter() - t0
        self.total_cpus = int(ray.cluster_resources().get("CPU", 0))
        self.ray_nodes = [n for n in ray.nodes() if n.get("Alive")]
        self.ray_address = address

        log("=" * 60)
        log(f"Ray address : {address}{' (local mode)' if self.local_mode else ''}")
        log(f"Ray nodes   : {len(self.ray_nodes)}")
        log(f"Ray CPUs    : {self.total_cpus}")
        log(f"Connect time: {self.init_time:.3f} s")
        log("=" * 60)
        if self.total_cpus <= 0:
            raise RuntimeError("Connected to Ray, but Ray reports zero CPUs.")
        return self.total_cpus, self.ray_nodes, self.init_time

    def stop_ray_clusters(self):
        """Disconnect; tear down the cluster only if this agent created it.
        In local mode, ray.shutdown() also stops the locally started instance."""
        if ray.is_initialized():
            ray.shutdown()
        if self.local_mode:
            self.local_mode = False
            log("Local Ray instance stopped")
            return
        if not self._owns_cluster:
            return
        for node in self.nodes:
            subprocess.run(["srun", "--overlap", "-N1", "-n1", "-w", node, "ray", "stop"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
        self._procs.clear()
        self._owns_cluster = False
        log("Ray cluster stopped")

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------
    def run_chunk(self, circuits, shots):
        """Submit one chunk; returns an ObjectRef -> (results, task_stats)."""
        return _run_chunk_remote.options(scheduling_strategy="SPREAD").remote(circuits, shots)

    @staticmethod
    def chunked(items, chunk_size):
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]

    def run_parallel(self, subexperiments, shots, total_cpus=None, task_multiplier=4):
        """Submit all subexperiments to Ray.

        Returns (results, info):
          results : {label: PrimitiveResult}  (order preserved; feed to reconstruct_expectation_values)
          info    : benchmark dict (timings, per-task, per-fragment, per-node stats)
        """
        total_cpus = total_cpus or self.total_cpus
        labels = list(subexperiments.keys())
        per_label_counts = {str(l): len(subexperiments[l]) for l in labels}
        total_circuits = sum(per_label_counts.values())

        if total_circuits == 0:
            info = {"num_tasks": 0, "chunk_size": 0, "total_circuits": 0,
                    "est_cpu_utilization": 0.0, "tasks": [], "per_node": {}, "per_fragment": {},
                    "timings": {"submit": 0.0, "execute_and_gather": 0.0, "wall": 0.0}}
            return {l: PrimitiveResult([]) for l in labels}, info

        target_tasks = total_cpus * task_multiplier
        chunk_size = max(1, math.ceil(total_circuits / target_tasks))

        # ---- submit ----
        t_wall0 = time.perf_counter()
        submitted = [
            (label, len(chunk), self.run_chunk(chunk, shots))
            for label in labels
            for chunk in self.chunked(list(subexperiments[label]), chunk_size)
        ]
        t_submit = time.perf_counter() - t_wall0
        num_tasks = len(submitted)
        #log(f"Submitted {total_circuits} circuits as {num_tasks} Ray tasks "
        #    f"(chunk size {chunk_size}, {total_cpus} CPUs) in {t_submit:.3f} s")

        # ---- execute + gather ----
        t1 = time.perf_counter()
        outputs = ray.get([f for _, _, f in submitted])   # order-preserving
        t_exec = time.perf_counter() - t1
        t_wall = time.perf_counter() - t_wall0

        # ---- assemble ----
        results = {l: [] for l in labels}
        tasks = []
        for (label, n, _), (res, stats) in zip(submitted, outputs):
            results[label].extend(res)
            stats = dict(stats, label=str(label))
            tasks.append(stats)

        # ---- benchmarks ----
        durations = [t["duration"] for t in tasks]
        busy_total = sum(durations)

        per_node = defaultdict(lambda: {"tasks": 0, "circuits": 0, "busy_seconds": 0.0})
        per_frag = defaultdict(lambda: {"tasks": 0, "circuits": 0, "busy_seconds": 0.0})
        for t in tasks:
            for d, key in ((per_node, t["host"]), (per_frag, t["label"])):
                d[key]["tasks"] += 1
                d[key]["circuits"] += t["n_circuits"]
                d[key]["busy_seconds"] += t["duration"]

        info = {
            "num_tasks": num_tasks,
            "chunk_size": chunk_size,
            "total_circuits": total_circuits,
            "target_tasks": target_tasks,
            "shots": shots,
            "total_cpus": total_cpus,
            "est_cpu_utilization": min(1.0, num_tasks / total_cpus),
            "timings": {
                "submit": t_submit,
                "execute_and_gather": t_exec,
                "wall": t_wall,
            },
            "measured_cpu_efficiency": busy_total / (t_wall * total_cpus) if t_wall > 0 else float("nan"),
            "speedup_vs_serial": busy_total / t_wall if t_wall > 0 else float("nan"),
            "circuits_per_second": total_circuits / t_wall if t_wall > 0 else float("nan"),
            "task_duration": {
                "min": min(durations), "max": max(durations),
                "mean": busy_total / len(durations), "sum": busy_total,
            },
            "nodes_used": len(per_node),
            "per_node": dict(per_node),
            "per_fragment": dict(per_frag),
            "per_fragment_circuits": per_label_counts,
            "tasks": tasks,
        }

        #log(f"Done in {t_wall:.2f} s | {info['circuits_per_second']:.1f} circuits/s | "
        #    f"efficiency {info['measured_cpu_efficiency']:.1%} | nodes used {len(per_node)}")

        return {l: PrimitiveResult(v) for l, v in results.items()}, info