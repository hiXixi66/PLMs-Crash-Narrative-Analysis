import matplotlib.pyplot as plt

# 数据
noise_ratios = [0, 5, 10, 15, 20, 30, 40, 50]

# LLaMA3-3B
llama_acc_excl9 = [0.964, 0.9573, 0.9535, 0.9506, 0.9457, 0.9291, 0.7941, None]
llama_f1_excl9 = [0.753, 0.7471, 0.7348, 0.7253, 0.7196, 0.6713, 0.5147, None]
llama_acc = [0.951, 0.9451, 0.9400, 0.9369, 0.9311, 0.9151, 0.7811, None]
llama_f1 = [0.779, 0.7792, 0.7526, 0.7391, 0.7249, 0.6748, 0.5085, None]

# BERT
bert_acc_excl9 = [0.929, 0.9248, 0.9190, 0.9169, 0.9132, 0.9013, 0.8893, 0.8678]
bert_f1_excl9 = [0.637, 0.6246, 0.6559, 0.6187, 0.6403, 0.5619, 0.5481, 0.5909]
bert_acc = [0.943, 0.9117, 0.9040, 0.9020, 0.8983, 0.8869, 0.8751, 0.8537]
bert_f1 = [0.620, 0.6482, 0.5559, 0.5238, 0.5426, 0.4762, 0.5462, 0.5006]

# 绘制 Accuracy 图
plt.figure(figsize=(8,6))
plt.plot(noise_ratios, llama_acc, marker='o', label="LLAMA3-3B Accuracy")
plt.plot(noise_ratios, bert_acc, marker='o', label="BERT Accuracy")
plt.xlabel("Noise Ratio (%)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Noise Ratio")
plt.legend()
plt.grid(True)
plt.show()

# 绘制 Macro F1 图
plt.figure(figsize=(8,6))
plt.plot(noise_ratios, llama_f1, marker='o', label="LLAMA3-3B Macro F1")
plt.plot(noise_ratios, bert_f1, marker='o', label="BERT Macro F1")
plt.xlabel("Noise Ratio (%)")
plt.ylabel("Macro F1")
plt.title("Macro F1 vs Noise Ratio")
plt.legend()
plt.grid(True)
plt.show()
