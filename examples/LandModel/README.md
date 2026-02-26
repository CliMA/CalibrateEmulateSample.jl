# Requirements of the script
Versioning (of when files were saved):
- Julia:
- JLD2:
- EnsembleKalmanProcesses:

Files:
- A JLD2 file containing the EKP object saved "ekp.jld2"
- A TOML file containing the prior "priors.toml"

Settables in the scrip
- The names of the parameters in the priors file, that we want to fit (optional) see top of script
- The number of iterations of EKI for training (the rest are used in testing the emulator)
- The type of dimension reduction (see docs), or one can just set the amount of truncation to do with the retain_var keywords
- Very brief evaluation on test set is performed. Proper validation should be completed by the scientist (Very important)
- Likewise, we plot the posterior, but other ways to validate this is left to the scientist. (Very important)


