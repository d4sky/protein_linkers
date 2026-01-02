### 🚀Usage
Edit the bottom section of build_linker.py to specify your input file and chain identifiers:

```python 
structure_file_path = "your_protein.pdb"
chainC = 'B'  # The growing chain (C-terminus)
chainN = 'A'  # The target chain (N-terminus)
nResis = 25   # Number of residues in the linker

mol_with_liner = find_linker(structure_file_path, chainC, chainN, nResis)
```

> [!IMPORTANT]
> ### 📝 Note on Input Structures
> The script requires a complete set of backbone atoms (**N, CA, C, O**) at the terminal residues of the chains being linked. 
>
> If your structure has missing terminal atoms, the script will print a termination message. As an example, the file `2I4Q_e.pdb` is provided in the repository; the Oxygen (**O**) atom was manually added to this file to ensure functionality.

### ⚙️ Parameters

The `find_linker` function accepts several parameters to fine-tune the search:

* **nRes (default: 15):** Sets the desired length of the linker in residues.
* **level (default: 1):** Determines the detail level of the structural assembly.
* **colThr (default: 0.60):** The energy/collision cutoff. If the script can't find a path, try increasing this to `1.2` or higher to allow for a tighter fit.
* **incr (default: 0.1):** How quickly the search expands the ellipsoid perimeter if a straight line is blocked.

