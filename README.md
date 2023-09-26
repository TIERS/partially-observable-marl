
<div align="center">
    <h1>A Study of Partial Observability in <br/> Multi-Agent Reinforcement Learning</h1>
    <a href="https://github.com/TIERS/partially-observable-marl/blob/main/LICENSE"><img src="https://img.shields.io/github/license/PRBonn/kiss-icp" /></a>
    <a href="https://github.com/TIERS/partially-observable-marl/blob/main"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/TIERS/partially-observable-marl/blob/main"><img src="https://img.shields.io/badge/Windows-0078D6?st&logo=windows&logoColor=white" /></a>
    <a href="https://github.com/TIERS/partially-observable-marl/blob/main"><img src="https://img.shields.io/badge/mac%20os-000000?&logo=apple&logoColor=white" /></a>
    <br />
    <br />
    <a href="#">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://github.com/TIERS/partially-observable-marl/issues">Contact Us</a>
  <br />
  <br />
  <p align="center">
    <!-- <img src="doc/setup.png" width=99% /> -->
  <div class="container">
    <img src="./results/MPE/simple_spread/rmappo/models/obs_2/render_20230920210334.gif" alt="drawing" width=20%/>
    <!-- <figcaption>2 agents.</figcaption> -->
    <img src="./results/MPE/simple_spread/rmappo/models/obs_4/render_20230920210235.gif" alt="drawing" width=20%/>
    <!-- <figcaption>4 agents.</figcaption> -->
    <img src="./results/MPE/simple_spread/rmappo/models/obs_6/render_20230920210207.gif" alt="drawing" width=20%/>
    <!-- <figcaption>6 agents.</figcaption> -->
    <img src="./results/MPE/simple_spread/rmappo/models/obs_8/render_20230920210043.gif" alt="drawing" width=20%/>
    <figcaption>Simple-Spread task: Agents with different partial observation settings can achieve comparable performance with near-optimality. From left to right: agents can observe nearby 2, 4, 6, 8 agents.</figcaption>
  </div>

  </p>

</div>

<!-- <div class="row">
  <div class="column">
    <img src="./results/MPE/simple_spread/rmappo/models/obs_2/render_20230920210334.gif" alt="drawing" width=20%/>
    <figcaption>obs=2.</figcaption>
  </div>
  <div class="column">
    <img src="./results/MPE/simple_spread/rmappo/models/obs_4/render_20230920210235.gif" alt="drawing" width=20%/>
    <figcaption>obs=4.</figcaption>
  </div>
  <div class="column">
    <img src="img_mountains.jpg" alt="Mountains" style="width:100%">
  </div>
  </div> -->

## Installation

```bash
$ conda env create -f environment.yml
```

## Train the agents

```bash
$ cd scripts
$ ./run_mpe_batch.sh
```

## Results

The pretrained simple-spread models can be found in `results/MPE/simple_spread/ramppo/models`
```bash
$ cd scripts
$ ./render_mpe.sh
```
## Citation

If you use this dataset for any academic work, please cite the following publication:

```
@misc{wenshuai2023less,
    title={Less Is More: Robust Robot Learning via Partially Observable Multi-Agent Reinforcement Learning}, 
    author={Wenshuai Zhao and Eetu Rantala and Joni Pajarinen and Jorge Peña Queralta},
    year={2023},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```
