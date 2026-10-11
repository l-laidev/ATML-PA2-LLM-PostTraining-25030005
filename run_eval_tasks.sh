trap 'exit 130' INT

echo 'DPO Evaluations'
uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/standard_model --name standard

uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/length_balanced/length_balanced_model --name length_balanced

uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/beta-0.03/beta-0.03_model --name beta-0.03
uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/beta-0.10/beta-0.10_model --name beta-0.10
uv run python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard/beta-0.30/beta-0.30_model --name beta-0.30


echo 'PPO Evaluations'
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/standard-policy --name standard

uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/kl-study/kl-beta-0.0000/kl-beta-0.0000-policy --name beta-0.0
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/kl-study/kl-beta-0.1000/kl-beta-0.1000-policy --name beta-0.1
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/kl-study/kl-beta-0.2000/kl-beta-0.2000-policy --name beta-0.2

uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/clipping-study/clip-0.0500/clip-0.0500-policy --name clip-0.05
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/clipping-study/clip-0.2000/clip-0.2000-policy --name clip-0.20
uv run python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/clipping-study/clip-0.5000/clip-0.5000-policy --name clip-0.50


echo 'GRPO Evaluations'
uv run python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard/standard-best --name standard

uv run python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard/norm-study/type-grpo --name grpo-norm
uv run python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard/norm-study/type-dr-grpo --name dr-grpo-norm


echo 'Safety Evaluations'
uv run python -m task4_safety.evaluate_safety --config configs/feedback.yaml
