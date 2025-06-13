# Source of rejected/chosen completions
python3 -m analysis.draw_model_histogram source_completion.pdf --log_scale --figsize 8 14
python3 -m analysis.draw_model_histogram source_completions_rejected_hori.pdf --log_scale --figsize 8 7 --top_n 20 --keys rejected_model
python3 -m analysis.draw_model_histogram source_completions_chosen_hori.pdf --log_scale --figsize 8 7 --top_n 20 --keys chosen_model
# Number of chosen and rejected per subset
python3 -m analysis.draw_subtoken_statistics prompt_length.pdf --figsize 16 10
# Violin plot of subset score distribution
python3 -m analysis.plot_per_subset_dist
# Plot per model
python3 -m analysis.plot_per_model_dist.py
# Make tables
python3 -m analysis.get_benchmark_results --render_latex
