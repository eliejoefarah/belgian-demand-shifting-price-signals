# belgian-demand-shifting-price-signals
Developing a signal to feed into an EMS that will allow for the demand of electricity to be shifted to when renewables are more available.

Title: How can the electricity demand of electrified Belgian dwellings be shifted to reduce the use of gas power plants?

Reducing our reliance on fossil fuels is essential to limit climate change. Electrifying our transport and heating systems, along with using clean electricity, are key solutions.

Unlike fossil fuels, solar and wind power are intermittent. This means electricity is only generated when the sun shines or the wind blows. This thesis will investigate how dwellings equipped with heat pumps, electric water heaters, and electric vehicle chargers can shift their electricity use to better match the production from clean energy sources.

This thesis builds on a previous study, "Should Belgians Install an Advanced Energy Management System in Their Electrified Dwelling?" ¹, which created a simulation of 500 representative electrified Belgian dwellings. 

The main goal of the thesis is to develop an algorithm that provides price signals for these dwellings. This will encourage them to shift the energy use of their heat pumps, water heaters, and electric vehicle chargers to times when clean energy is abundant and reduce the need for gas power plants. The algorithm must account for the uncertainty in renewable energy forecasts.

¹Not published yet but content available in Chapter 7 of https://www.researchgate.net/profile/Julian-Ruddick/publication/401851512_How_to_Effectively_Manage_the_Energy_in_Residential_Systems_Development_and_Comparison_of_Methods/links/69b28e4ac325e338e8e8b8a7/How-to-Effectively-Manage-the-Energy-in-Residential-Systems-Development-and-Comparison-of-Methods.pdf
## Getting the code

This repo includes Julian Ruddick's paper code as a git submodule, `should_belgians_install_advanced_ems_paper/`, pinned to commit `d8338ce`. Clone both in one go:

```bash
git clone --recurse-submodules https://github.com/eliejoefarah/belgian-demand-shifting-price-signals.git
```

The paper repo has its own submodule (OpenStudio-HPXML) with an SSH URL. That submodule is not used here, so if `--recurse-submodules` fails without a GitHub SSH key, clone without it and fetch only the paper repo:

```bash
git clone https://github.com/eliejoefarah/belgian-demand-shifting-price-signals.git
cd belgian-demand-shifting-price-signals
git submodule update --init should_belgians_install_advanced_ems_paper
```

Three things are not in any git repo and must be set up by hand. Follow these steps of `should_belgians_install_advanced_ems_paper/README.md`:
- **Step 3:** Gurobi licence.
- **Step 4:** EnergyPlus 24.1.0 archive.
- **Step 5:** the Zenodo download of `data/` and `treec_train_500/` (about 22 GB; record 18417982).

Then create the environment below.

## Environment setup

The environment is frozen in `environment.yml` (conda + pip, Python 3.13). To rebuild it from the repo root:

```bash
conda env create -f environment.yml
conda activate belgian_dwellings
pip install --no-build-isolation --no-deps should_belgians_install_advanced_ems_paper/ \
    should_belgians_install_advanced_ems_paper/submodules/treec/ \
    should_belgians_install_advanced_ems_paper/submodules/energy-system-simulation/
pip install -e .
```

The last two lines install the local packages (`belgian_dwellings`, `treec`, `simugrid`, and this repo's `fleet_harness`/`price_signals`), which are not on PyPI. EnergyPlus 24.1.0 and the Gurobi licence are set up as described in `should_belgians_install_advanced_ems_paper/README.md`.

### Keeping `environment.yml` up to date

Re-export the environment every time you **install, upgrade or remove a package**, whether with conda or pip. Editing your own code (`fleet_harness/`, `price_signals/`, the paper repo) does not need a re-export.

One-time setup: add this alias to `~/.zshrc`, then run `source ~/.zshrc`:

```bash
alias freeze-env='conda env export -n belgian_dwellings --no-builds | sed -E "/^\s*- (belgian-dwellings|fleet-harness|simugrid|treec)==/d; /^prefix:/d" > environment.yml'
```

Each time:

1. From the **repo root** (not inside `should_belgians_install_advanced_ems_paper/`), run `freeze-env`.
2. Check the change with `git diff environment.yml`.
3. Commit `environment.yml`.

The alias does the same as a plain `conda env export`, but also strips the lines that would break a rebuild:
- the local packages `belgian-dwellings`, `fleet-harness`, `simugrid` and `treec` (not on PyPI; installed from source above);
- the machine-specific `prefix:` line.
