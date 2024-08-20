import math
import json
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from utils.transformations import transform_tensor_batch
from utils.local_search import tsp_length_batch, combined_local_search
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
from problems.tsp.problem_tsp import TSPDataset
mp = torch.multiprocessing.get_context('spawn')
import pickle as pkl


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    use_oracle = opts.oracle_baseline is not None
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
            
        if not opts.load_tsplib:
            # Case 1: Load dataset from test file
            dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
            
            # Load oracle if necessary
            if use_oracle:
                with open(opts.oracle_baseline, 'rb') as f:
                    oracle_baseline = pkl.load(f)
                assert len(oracle_baseline) == len(dataset), "Oracle baseline does not have same number of entries as dataset"
            
            # Evaluate model on dataset
            results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)
        else:
            # Case 2: Load dataset from TSPLib folder
            names = []
            for filename in os.listdir(dataset_path):
                if filename.endswith(".npy"):
                    if not filename.endswith("sol.npy"):
                        names.append(filename.split(".")[0])
            
            use_oracle = True
            results = []
            oracle_baseline = []
            for name in names:
                dataset = model.problem.make_dataset(
                    filename=os.path.join(dataset_path, f"{name}.npy"), num_samples=1, offset=0
                )
                oracle_baseline.append(np.load(os.path.join(dataset_path, name + "_sol.npy"))[0][0][0])
                results.extend(_eval_dataset(model, dataset, width, softmax_temp, opts, device))

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    # Print gap-based stats if applicable
    gap_rel = None
    if use_oracle and oracle_baseline is not None:
        oracle_costs = np.array(oracle_baseline)
        gap = costs - oracle_costs
        gap_rel = gap / oracle_costs * 100
        print("Average relative gap: {}% +- {}%".format(np.mean(gap_rel), 2 * np.std(gap_rel) / np.sqrt(len(gap_rel))))
        print("Gap stats: min {}, max {}, mean {}, std {}".format(
            np.min(gap), np.max(gap), np.mean(gap), np.std(gap)
        ))

    # Print general stats
    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
    if opts.o is None:
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))
    else:
        out_file = opts.o

    if not opts.all_epochs:
        print(out_file)
        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        save_dataset((results, gap_rel), out_file)

    return costs, gap_rel, tours, durations, os.path.join(results_dir, model_name)


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)


    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        
        batch = move_to(batch, device)
        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                if opts.data_equivariance:
                    batch = transform_tensor_batch(batch)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                if opts.local_search:
                    combined_local_search(batch, sequences)
                    costs = tsp_length_batch(batch, sequences)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")

    parser.add_argument('--load_tsplib', action='store_true', help="Whether to treat the input path as a TSPLib folder")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--data_equivariance', action='store_true',
                        help='Apply rotational and translational invariance during evaluation')
    parser.add_argument('--local_search', action='store_true',
                        help='Apply a local search to optimize found paths')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--oracle_baseline', type=str, default=None, help='Oracle baseline for computing gap statistics')
    parser.add_argument('--model', type=str)
    parser.add_argument('--all_epochs', action='store_true', help='Evaluate all epochs')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    assert not (opts.all_epochs and opts.model.endswith(".pt")), "Can only use --all_epochs on a folder"

    widths = opts.width if opts.width is not None else [0]

    for width in widths:
        for dataset_path in opts.datasets:
            if opts.all_epochs:
                base_model_path = opts.model
                res = {}
                for epoch_file in os.listdir(opts.model):
                    if not epoch_file.endswith(".pt"):
                        continue
                    epoch = int(epoch_file.split("-")[1].split(".")[0])
                    model_path = os.path.join(base_model_path, epoch_file)
                    opts.model = model_path
                    costs, gap_rel, _, _, results_prefix = eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
                    percent_50 = len(costs) // 2
                    percent_1 = len(costs) // 100
                    percent_05 = len(costs) // 200
                    percent_01 = len(costs) // 1000
                    res[str(epoch)] = {
                        "Cost_Avg": str(np.mean(costs)),
                        "Cost_Error": str(2 * np.std(costs) / np.sqrt(len(costs))),
                        "Gap_Avg": str(np.mean(gap_rel) if gap_rel is not None else -1),
                        "Gap_Error": str(2 * np.std(gap_rel) / np.sqrt(len(gap_rel)) if gap_rel is not None else -1),
                        "Gap_Best_50": str(np.mean(gap_rel[np.argsort(gap_rel)[:percent_50]])),
                        "Gap_Worst_1.0": str(np.mean(gap_rel[np.argsort(gap_rel)][-percent_1:])),
                        "Gap_Worst_0.5": str(np.mean(gap_rel[np.argsort(gap_rel)][-percent_05:])),
                        "Gap_Worst_0.1": str(np.mean(gap_rel[np.argsort(gap_rel)][-percent_01:]))
                    }
                    results_prefix = results_prefix[:-len(str(epoch)+'_epoch-')]
                with open(results_prefix + "-epoch_data.json", 'w') as f:
                    json.dump(res, f, indent=2)
            else:
                eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
