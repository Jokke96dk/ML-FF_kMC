import os
import pickle


system_data = {}
base_filename = "data_MLFF"
i = 1
while True:
    filename = f"copy_{base_filename}_{i}.pkl"
    if not os.path.exists(filename):
        break
    with open(filename, "rb") as f:
        chunk = pickle.load(f)
        system_data.update(chunk)
    print(f"Loaded {len(chunk)} items from {filename}")
    i += 1


from ase import Atoms

# ---- Loop 1: create framework once ----
framework_atoms = None

for key, value in system_data.items():
    atoms = value[1]
    symbols = atoms.get_chemical_symbols()

    mask_framework = [s in ("O", "Si", "F", "Ne") for s in symbols]
    candidate = atoms[mask_framework]

    # store the FIRST framework we find
    if len(candidate) > 0:
        framework_atoms = candidate
        break

# Now add framework to the dictionary (outside loops!)
system_data["framework"] = [None, framework_atoms]


# ---- Loop 2: remove O, Si, F from all NON-framework structures ----
for key, value in list(system_data.items()):
    if key == "framework":
        continue

    atoms = value[1]
    symbols = atoms.get_chemical_symbols()

    mask_non_framework = [s not in ("O", "Si", "F","Ne") for s in symbols]
    value[1] = atoms[mask_non_framework]
    system_data[key] = value


base_filename="data_MLFF"
chunk_size = 100000 # Keep this many entries per file
items = list(system_data.items())
#n_chunks = math.ceil(len(items) / chunk_size)
n_chunks = (len(items) + chunk_size - 1) // chunk_size
for i in range(n_chunks):
    chunk_items = items[i*chunk_size:(i+1)*chunk_size]
    chunk_dict = dict(chunk_items)
    filename = f"{base_filename}_{i+1}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(chunk_dict, f)
    print(f"Saved {len(chunk_dict)} items to {filename}")

