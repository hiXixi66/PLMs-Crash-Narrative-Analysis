import json
import matplotlib.pyplot as plt

# 文件路径（你提供的路径）
# trainer_state_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/qwen2.5-finetune-crashtypeqkv/checkpoint-2163/trainer_state.json"
trainer_state_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/llama3B-ft-MANCOLL-noise-test/with-bias-30perc/checkpoint-1668/trainer_state.json"
# 尝试加载文件
try:
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)
except FileNotFoundError:
    trainer_state = None

# 检查是否成功加载
if trainer_state and "log_history" in trainer_state:
    logs = trainer_state["log_history"]

    # 提取有效的训练日志项（包含 step 和 loss 等）
    steps, losses, learning_rates, grad_norms = [], [], [], []
    for entry in logs:
        if "step" in entry and "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
            learning_rates.append(entry.get("learning_rate", None))
            grad_norms.append(entry.get("grad_norm", None))

    # 绘图
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(steps, losses, marker='o')
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss")

    axs[1].plot(steps, learning_rates, marker='o', color='green')
    axs[1].set_ylabel("Learning Rate")
    axs[1].set_title("Learning Rate Schedule")

    axs[2].plot(steps, grad_norms, marker='o', color='red')
    axs[2].set_ylabel("Gradient Norm")
    axs[2].set_title("Gradient Norm")
    axs[2].set_xlabel("Training Step")

    plt.tight_layout()
    output_path = "reports/crash-type-test/figure/training_metrics_from_json2.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    result = output_path
else:
    result = "Failed to load or parse trainer_state.json. Please check the file path or contents."

result
