CUDA_VISIBLE_DEVICES=0 python /data/szhang967/dlm_agents/sudoku/eval/sweep_eval.py \
  --checkpoint /data/szhang967/dlm_agents/sudoku/absorbing_checkpoints/best_model.pt \
  --data-dir /data/szhang967/dlm_agents/sudoku/data \
  --device cuda \
  --project sudoku_eval_new1 \
  --output-dir /data/szhang967/dlm_agents/sudoku/eval_results_new1 \
  --eval-batch-size 2000 \
  --num-samples 2000 \
  --modes absorbing uniform_noise_only uniform_noise_diffusion