trap 'exit 130' INT
set -e

echo 'DPO Tasks'
# uv run python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
# uv run python -m task1_dpo.ablate_beta --config configs/dpo.yaml
# uv run python -m task1_dpo.analyze_length --config configs/dpo.yaml

echo 'PPO Tasks'
# uv run python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
# uv run python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
# uv run python -m task2_ppo.ablate_kl --config configs/ppo.yaml

echo 'GRPO Tasks'
# uv run python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
# uv run python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
# uv run python -m task3_grpo.compare_normalization --config configs/grpo.yaml

echo 'Safety Tasks'
# uv run python -m task4_safety.generate_responses --config configs/feedback.yaml
# uv run python -m task4_safety.judge_responses --config configs/feedback.yaml
# uv run python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
# uv run python -m task4_safety.join_responses --config configs/feedback.yaml

echo 'Feedback Tasks'
# uv run python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
# uv run python -m task5_feedback.score_perturbations --config configs/feedback.yaml
# uv run python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
# uv run python -m task5_feedback.compare_feedback --config configs/feedback.yaml