# Model Predictive Control and Moving Horizon Estimation using Statistically Weighted Data-Based Ensemble Models

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/dhs-ensemble/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.13-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Model Predictive Control and Moving Horizon Estimation using Statistically Weighted Data-Based Ensemble Models](https://arxiv.org/abs/2511.21343) submitted to [ECC 2026](https://ecc26.euca-ecc.org/).

In this work we propose a control and estimation framework, based on ensemble models, for the optimal control of complex systems under multiple operating conditions.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{de2025model,
  title={Model Predictive Control and Moving Horizon Estimation using Statistically Weighted Data-Based Ensemble Models},
  author={de Giuli, Laura Boca and Mallick, Samuel and La Bella, Alessio and Dabiri, Azita and De Schutter, Bart and Scattolini, Riccardo},
  journal={arXiv preprint arXiv:2511.21343},
  year={2025}
}
```

---

## Installation

The code was created with `Python 3.13`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/dhs-ensemble
cd dhs-ensemble
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`mpc`** contains the class for the MPC controller.
- **`observer`** contains the classes for open loop and moving horizon state estimators.
- **`plotting`** contains the scripts for generating all images in the paper: Model Predictive Control and Moving Horizon Estimation using Statistically Weighted Data-Based Ensemble Models.
- **`prediction_model`** contains the model weight files, as mat files, and the class for the RNN prediction model.
- **`results`** contains all data, as pickles, for the results in the paper: Model Predictive Control and Moving Horizon Estimation using Statistically Weighted Data-Based Ensemble Models.
- **`sim_data`** contains the data for running different simulations, e.g., load profiles and electricity prices.
- **`simulation_model`** contains the functional mockup unit used as a high-fidelity simuator.
- **`env.py`** contains the environment class that steps the simulation model and generates feedback signals for the controllers.
- **`run_mpc.py`** is the main script for running all simulations.
## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/dhs-ensemble/blob/main/LICENSE) file included with this repository.

---

## Author
[Laura Boca de Giuli](https://www.deib.polimi.it/ita/personale/dettagli/1210274), PhD Candidate [laura.bocadegiuli@polimi.it]

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> [Dipartimento di Elettronica, Informazione e Bioingegneria](https://www.deib.polimi.it/ita/home) in [Politecnico di Milano](https://www.polimi.it/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

> This research is part of a project that has received fundtion from Next-Generation EU
(Italian PNRR - M4 C2, Invest 1.3 - D.D. 1551.11-10-2022,
PE00000004). CUP MICS D43C22003120001.

Copyright (c) 2024 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “dhs-ensemble” (Model Predictive Control and Moving Horizon Estimation using Statistically Weighted Data-Based Ensemble Models) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.