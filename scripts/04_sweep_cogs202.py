# Sweeps the `cogs202_result` DVs (figure3_anns.ipynb) across regimes x noise levels.
#
# Per cell it records three metrics for the `--condition` schedule:
#   t            - timestep at which the smoothed Task-B loss first comes within
#                  0.001 of its 5th-percentile value (time to near-asymptotic loss).
#   learn_auc    - area below the post-transfer baseline during Task A1 (how much
#                  was learned).
#   transfer_auc - area above the post-transfer baseline during Task B (interference
#                  cost while transferring).
#
# Default grid: regimes {lazy_50, rich_50} x full factorial A_sd,B_sd in {0,8}.
# Missing (regime, noise) cells are trained on demand (after confirmation).
#
# Only the `--condition` schedule (default: near) feeds the DV, so the sweep
# trains just those participants -- the schedule is encoded in the participant
# name -- which cuts ~2/3 of the work versus training same/near/far.
#
# Training is parallelized over a FLAT list of (cell x participant) jobs: every
# missing cell contributes its participants to one shared process pool, so cells
# and participants run concurrently and load-balance across cores. Each
# participant is seeded by its index (2024 + idx) so results are reproducible
# regardless of how jobs are scheduled.
#
#   python scripts/04_sweep_cogs202.py                  # full sweep, prompts before training
#   python scripts/04_sweep_cogs202.py --jobs 8         # cap the pool at 8 workers
#   python scripts/04_sweep_cogs202.py --yes            # train missing cells without prompting
#   python scripts/04_sweep_cogs202.py --force          # retrain even cells that already exist
#   python scripts/04_sweep_cogs202.py --compute-only   # only score existing cells
#   python scripts/04_sweep_cogs202.py --regimes rich_50 --a-sds 0 --b-sds 0

import os

# Pin BLAS/OpenMP to one thread per worker process so the pool isn't fighting
# itself. The model is tiny, so single-thread workers + many processes wins.
# Must be set before torch / numpy import their backends.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import argparse
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Project root discovery (mirrors notebooks/figure3_anns.ipynb cell 1)
project_root = Path(__file__).resolve().parent
while not (project_root / 'src').exists():
    if project_root == project_root.parent:
        raise RuntimeError("Project root directory not found.")
    project_root = project_root.parent
sys.path.append(str(project_root))

from src.analysis import ann
from src.models import neural_network as net

SEED_BASE = 2024        # per-participant seed = SEED_BASE + participant_index


# --- worker process state (populated once per process by the initializer) ------
_DF = None
_TASKP = None


def _init_worker(base_folder):
    """Run once per worker: pin threads and load the (read-only) shared data."""
    import torch
    torch.set_num_threads(1)
    global _DF, _TASKP
    _DF = ann.load_participant_data(os.path.join(base_folder, 'data'))
    _TASKP = ann.setup_task_parameters()


def _train_job(job):
    """Train + save one participant for one cell. Args are all picklable primitives."""
    condition, a_sd, b_sd, participant, seed, sim_folder, n_epochs, n_phase, shuffle, batch_size, lr = job

    dim_input = _TASKP['nStim_perTask'] * 2
    network_params = [dim_input, condition['dim_hidden'], 4]
    n_train_trials = n_epochs * dim_input * 10
    training_params = [None, n_phase, n_epochs, n_train_trials, shuffle, batch_size, condition['gamma'], lr]

    res = net.train_one_participant(
        participant, _DF, _TASKP, network_params, training_params,
        do_test=1, a_error_sd=a_sd, b_error_sd=b_sd, seed=seed,
    )
    # Write atomically: a reader (or a second sweep) must never see a partial
    # archive. savez to a temp file in the same dir, then os.replace.
    final = os.path.join(sim_folder, f'sim_{participant}.npz')
    tmp = os.path.join(sim_folder, f'.sim_{participant}.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **res)
    os.replace(tmp, final)
    return condition['name'], a_sd, b_sd, participant


# --- helpers -------------------------------------------------------------------
def sim_folder(base_folder, regime, a_sd, b_sd):
    return Path(base_folder) / 'data' / f'simulations_A-{a_sd}_B-{b_sd}' / regime


def cell_has_data(base_folder, regime, a_sd, b_sd):
    folder = sim_folder(base_folder, regime, a_sd, b_sd)
    return folder.is_dir() and any(f.endswith('.npz') for f in os.listdir(folder))


def write_settings(folder, condition, settings, task_parameters, participants, a_sd, b_sd):
    """Record run config alongside the npz files (mirrors 02_run_simulations.py)."""
    dim_input = task_parameters['nStim_perTask'] * 2
    to_save = {
        "condition": condition,
        "training_params": {
            "participants": ann.numpy_to_python(participants),
            "n_phase": settings['n_phase'],
            "n_epochs": settings['n_epochs'],
            "n_train_trials": settings['n_epochs'] * dim_input * 10,
            "shuffle": settings['shuffle'],
            "batch_size": settings['batch_size'],
            "gamma": condition['gamma'],
            "lr": settings['learning_rate'],
            "a_error_sd": a_sd,
            "b_error_sd": b_sd,
            "seed_base": SEED_BASE,
        },
        "network_params": [dim_input, condition['dim_hidden'], 4],
        "task_parameters": task_parameters,
    }
    with open(os.path.join(folder, 'settings.json'), 'w') as f:
        json.dump(ann.numpy_to_python(to_save), f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Sweep cogs202_result across regimes x noise levels')
    parser.add_argument('--base-folder', type=str, default=str(project_root))
    parser.add_argument('--regimes', nargs='+', default=['lazy_50', 'rich_50'])
    parser.add_argument('--a-sds', nargs='+', type=int, default=[0, 8])
    parser.add_argument('--b-sds', nargs='+', type=int, default=[0, 8])
    parser.add_argument('--condition', type=str, default='near',
                        help="Schedule the DV is computed on (default: near)")
    parser.add_argument('--jobs', type=int, default=os.cpu_count(),
                        help='Max worker processes for training (default: all cores)')
    parser.add_argument('--yes', action='store_true',
                        help='Train missing cells without prompting')
    parser.add_argument('--force', action='store_true',
                        help='Retrain cells even if their data already exists')
    parser.add_argument('--compute-only', action='store_true',
                        help='Only score existing cells; skip training, warn on missing')
    parser.add_argument('--out', type=str, default=None,
                        help='Output CSV path (default: <base>/data/cogs202_results.csv)')
    args = parser.parse_args()

    base = args.base_folder
    cells = [(r, a, b) for r in args.regimes
             for a, b in itertools.product(args.a_sds, args.b_sds)]

    present = [c for c in cells if cell_has_data(base, *c)]
    to_train = [c for c in cells if args.force or c not in present]

    print(f"Grid: {len(cells)} cells ({len(args.regimes)} regimes x "
          f"{len(args.a_sds) * len(args.b_sds)} noise levels)")
    print(f"  {len(present)} already have data"
          + (f", {len(to_train)} to (re)train" if not args.compute_only else ""))
    for r, a, b in to_train:
        print(f"    train: {r}  A={a} B={b}" + ("  (force-retrain)" if (r, a, b) in present else ""))

    trained = set()
    if to_train and not args.compute_only:
        # Load config + the fixed 20-participant sample (same scheme as 02_run_simulations.py).
        with open(os.path.join(base, 'src', 'models', 'ann_experiments.json')) as f:
            settings = json.load(f)
        conditions = {c['name']: c for c in settings['conditions']}
        task_parameters = ann.setup_task_parameters()
        df = ann.load_participant_data(os.path.join(base, 'data'))
        # Only the `--condition` schedule contributes to the DV, so train just
        # those participants (the schedule is encoded in the participant name,
        # e.g. study1_near_sub5). Cuts ~2/3 of the work for a near-only sweep.
        participants = [p for p in df['participant'].unique() if args.condition in str(p)]
        print(f"  {len(participants)} '{args.condition}' participants per cell")

        jobs = []
        for r, a, b in to_train:
            if r not in conditions:
                raise ValueError(f"Condition '{r}' not found in ann_experiments.json")
            cond = conditions[r]
            folder = sim_folder(base, r, a, b)
            folder.mkdir(parents=True, exist_ok=True)
            write_settings(str(folder), cond, settings, task_parameters, participants, a, b)
            for idx, participant in enumerate(participants):
                jobs.append((cond, a, b, str(participant), SEED_BASE + idx, str(folder),
                             settings['n_epochs'], settings['n_phase'],
                             settings['shuffle'], settings['batch_size'], settings['learning_rate']))

        n_workers = max(1, min(args.jobs, len(jobs)))
        print(f"\n{len(jobs)} participant-jobs across {len(to_train)} cell(s), "
              f"{n_workers} worker process(es).")
        if not args.yes:
            resp = input("Proceed with training? [y/N] ").strip().lower()
            if resp not in ('y', 'yes'):
                print("Aborted before training. Re-run with --compute-only to score "
                      "only existing cells.")
                return

        done = 0
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker, initargs=(base,)) as pool:
            futures = [pool.submit(_train_job, job) for job in jobs]
            for fut in as_completed(futures):
                rname, a, b, participant = fut.result()
                trained.add((rname, a, b))
                done += 1
                print(f"  [{done}/{len(jobs)}] {rname} A={a} B={b}  {participant}")

    rows = []
    participant_rows = []
    for r, a, b in cells:
        if not cell_has_data(base, r, a, b):
            print(f"  skipping (no data): {r}  A={a} B={b}")
            continue
        ann_data = ann.load_ann_data(str(sim_folder(base, r, a, b)))
        t = ann.compute_loss_time_to_pct(ann_data, condition=args.condition)
        learn_auc, transfer_auc = ann.compute_loss_auc(ann_data, condition=args.condition)
        tag = "trained" if (r, a, b) in trained else "reused"
        print(f"  {r}  A={a} B={b}  ({tag})  ->  t={t}  "
              f"learn_auc={learn_auc:.1f}  transfer_auc={transfer_auc:.1f}")
        rows.append({'regime': r, 'a_error_sd': a, 'b_error_sd': b, 't': t,
                     'learn_auc': learn_auc, 'transfer_auc': transfer_auc})

        for pr in ann.compute_loss_metrics_per_participant(ann_data, condition=args.condition):
            participant_rows.append({'regime': r, 'a_error_sd': a, 'b_error_sd': b, **pr})

    df_out = pd.DataFrame(rows, columns=['regime', 'a_error_sd', 'b_error_sd', 't',
                                         'learn_auc', 'transfer_auc'])
    print("\n" + df_out.to_string(index=False))

    out_path = Path(args.out) if args.out else Path(base) / 'data' / 'cogs202_results.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {out_path}")

    # Per-participant rows (the "original data" behind each cell's aggregate),
    # used for jittered scatter + bootstrapped CIs in 05_plot_cogs202.py.
    df_participants = pd.DataFrame(
        participant_rows,
        columns=['regime', 'a_error_sd', 'b_error_sd', 'participant', 't',
                 'learn_auc', 'transfer_auc'])
    participants_path = out_path.with_name(out_path.stem + '_participants.csv')
    df_participants.to_csv(participants_path, index=False)
    print(f"Saved {len(df_participants)} participant rows to {participants_path}")


if __name__ == "__main__":
    main()
