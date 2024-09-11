## Python Scripts

Scripts used for observations, historical simulations and picontrol simulation are split into three separate directories. [sbatch_job_submit](./sbatch_job_submit) is for submitting jobs on [JASMIN](https://jasmin.ac.uk). Below is a short description of different scripts.

| Script Name Structure | Content |
| --- | --- | 
| Compute_SST_Heat_Content | To compute time-series of heat content, SST, NAO |
| Composites_SST_NAO | To compute NAO-based composites | 
| Correlations | To compute correlations between observed and reconstructed temperatures |
| Parameter_tau_beta | To create temperature reconstuctions using NAO timeseries |
| Response_Function | Same as composites but created by substracting NAO- composites from NAO+ composites |
