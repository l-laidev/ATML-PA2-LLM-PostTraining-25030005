trap 'exit 130' INT
set -e

echo 'DPO Evaluations'
# uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/standard_model_final --name standard

# uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/length_balanced/length-balanced_model_final --name length_balanced

# uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/beta-0.03/beta-0.03_model_final --name beta-0.03
uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/beta-0.1/beta-0.1_model_final --name beta-0.10
uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/beta-0.3/beta-0.3_model_final --name beta-0.30


echo 'PPO Evaluations'
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/standard-final-policy --name standard

uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/kl-study/kl-beta-0.0000/kl-beta-0.0000-final-policy --name beta-0.0
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/kl-study/kl-beta-0.1000/kl-beta-0.1000-final-policy --name beta-0.1
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/kl-study/kl-beta-0.2000/kl-beta-0.2000-final-policy --name beta-0.2

uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/clipping-study/clip-0.0500/clip-0.0500-final-policy --name clip-0.05
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/clipping-study/clip-0.2000/clip-0.2000-final-policy --name clip-0.20
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/clipping-study/clip-0.5000/clip-0.5000-final-policy --name clip-0.50


echo 'GRPO Evaluations'
uv run python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard/standard-final --name standard

uv run python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard/norm-study/type-grpo-final --name grpo-norm
uv run python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard/norm-study/type-dr-grpo-final --name dr-grpo-norm


echo 'Safety Evaluations'
uv run python -m task4_safety.evaluate_safety --config configs/feedback.yaml
