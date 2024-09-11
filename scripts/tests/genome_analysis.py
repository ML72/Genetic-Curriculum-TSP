import numpy as np
import matplotlib.pyplot as plt
import os



# Define constants, model fields as necessary
TRIAL_DIR = 'outputs/tsp_50/test_verbose_1'
RESULT_FILENAME = 'genome_analysis_base{}.png'
GENOME_BASE_INCREMENTS = [
    1, 0.01, 0.1, 0.1, 0.1, 0.1
]

# Get all epochs from directory
epochs = list(set([
    int(os.path.splitext(filename)[0].split("-")[1].split("_")[0])
    for filename in os.listdir(TRIAL_DIR)
    if os.path.splitext(filename)[1] == '.npy'
]))
epochs.sort()
assert epochs, "No epochs found in {}".format(TRIAL_DIR)

# Read all numpy arrays
genomes = np.array([
    np.load(os.path.join(TRIAL_DIR, 'epoch-{}_genome.npy'.format(epoch)))
    for epoch in epochs
])

# Plot genome density
for base, inc in enumerate(GENOME_BASE_INCREMENTS):
    print(f"Plotting genome base {base} density")

    # Compute histograms
    base_slice = genomes[:,:,base]
    base_slice = inc * np.round(base_slice / inc)
    slice_min = np.min(base_slice)
    slice_max = np.max(base_slice)
    vals = np.linspace(
        slice_min-inc/2,
        slice_max+inc/2,
        num=2+round((slice_max-slice_min)/inc),
        endpoint=True
    )
    bins = np.zeros((len(epochs), len(vals)-1))
    for i in range(len(epochs)):
        bins[i], _ = np.histogram(base_slice[i], bins=vals)

    # Map to log scale for better visibility
    bins = np.log(bins+1)

    # Plot heatmap
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(vals)), minor=False)
    ax.set_yticks(np.arange(len(epochs))+0.5, minor=False)
    ax.set_xticklabels(np.round(vals, 2), rotation=90, fontsize=8)
    ax.set_yticklabels(epochs, fontsize=8)

    heatmap = ax.pcolor(bins, cmap='inferno')
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Genome Frequency (Log Scale)")
    ax.invert_yaxis()
    fig.set_figwidth(5)
    fig.set_figheight(8)

    plt.title(f"Genome Base {base} Frequency")
    plt.ylabel("Epoch")
    plt.xlabel(f"Base {base} Value Range")

    # Save plot
    if not os.path.exists("results/analysis"):
        os.makedirs("results/analysis")
    plt.savefig(f"results/analysis/{RESULT_FILENAME.format(base)}", format='png')
    plt.close()
