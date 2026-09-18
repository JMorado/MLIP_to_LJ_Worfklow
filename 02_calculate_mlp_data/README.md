# Calculate MLP data

- The `calculate_mlp_data.py` script should be copied into each `run_*` folder, for example:

```bash
for i in run_*/; do
  cp calculate_mlp_data.py "$i"
done
```

- `calculate_mlp_data.py` computes energies and forces.
- The script [submit_calculate_mlp_data.sh](submit_calculate_mlp_data.sh) can be used to run the calculations on `comet`. Note that these calculations are highly memory intensive.