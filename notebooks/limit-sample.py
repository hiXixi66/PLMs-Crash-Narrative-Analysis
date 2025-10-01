import matplotlib.pyplot as plt

# llama3B 实验结果
samples = [200, 400, 600, 800, 1200, 1600, 2000]

# LLaMA3-3B
llama_acc_excl9 = [x*100 for x in [0.3003,0.3308,0.8304,0.9158,0.9259,0.9466,0.9500]]
llama_f1_excl9 = [x*1 for x in [0.109,0.1450,0.5943,0.7651,0.7844,0.7148,0.7274]]
llama_acc_all  = [x*100 for x in [0.297,0.3254,0.8169,0.9009,0.9109,0.9323,0.9360]]
llama_f1_all   = [x*1 for x in [0.1152,0.1258,0.5037,0.6478,0.6631,0.7241,0.7398]]

# BERT
bert_acc_excl9 = [x*100 for x in [0.7380,0.8292,0.855,0.8713,0.9056,0.9239,0.9378]]
bert_f1_excl9  = [x*1 for x in [0.3067,0.4178,0.4350,0.5520,0.6327,0.6749,0.7482]]
bert_acc_all   = [x*100 for x in [0.7260,0.8157,0.8414,0.8571,0.8909,0.9089,0.9226]]
bert_f1_all    = [x*1 for x in [0.2606,0.3546,0.3691,0.4685,0.5361,0.5704,0.6320]]

# fastText
fasttext_acc_excl6 = [x*100 for x in [0.4615,0.5225,0.6876,0.6890,0.6882,0.7099,0.7346]]
fasttext_f1_excl6  = [x*1 for x in [0.1822,0.2028,0.2619,0.2637,0.2641,0.2892,0.3218]]
fasttext_acc_all   = [x*100 for x in [0.4541,0.5141,0.6765,0.6778,0.6770,0.6984,0.7227]]
fasttext_f1_all    = [x*1 for x in [0.1548,0.1725,0.2224,0.2241,0.2243,0.2457,0.2736]]



# 设置全局字体大小和线条粗细
plt.rcParams.update({"font.size": 14, "lines.linewidth": 2.5})

# 保存成 PDF 格式

# Accuracy 对比
plt.figure(figsize=(6,4.5))
plt.plot(samples, llama_acc_all, marker="o", color="tab:blue", label="LLaMA3B Accuracy (All)")
plt.plot(samples, llama_acc_excl9, marker="o", linestyle="--", color="tab:blue", label="LLaMA3B Accuracy (Excl. 9)")
plt.plot(samples, bert_acc_all, marker="s", color="tab:orange", label="BERT Accuracy (All)")
plt.plot(samples, bert_acc_excl9, marker="s", linestyle="--", color="tab:orange", label="BERT Accuracy (Excl. 9)")
plt.plot(samples, fasttext_acc_all, marker="^", color="tab:green", label="FastText Accuracy (All)")
plt.plot(samples, fasttext_acc_excl6, marker="^", linestyle="--", color="tab:green", label="FastText Accuracy (Excl. 6)")
plt.xlabel("Number of Training Samples", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(samples, [str(s) for s in samples])
plt.title("Accuracy Comparison", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("notebooks/accuracy_comparison_colored.pdf")
plt.show()

# Macro F1 对比
plt.figure(figsize=(6,4.5))
plt.plot(samples, llama_f1_all, marker="o", color="tab:blue", label="LLaMA3B Macro F1 (All)")
plt.plot(samples, llama_f1_excl9, marker="o", linestyle="--", color="tab:blue", label="LLaMA3B Macro F1 (Excl. 9)")
plt.plot(samples, bert_f1_all, marker="s", color="tab:orange", label="BERT Macro F1 (All)")
plt.plot(samples, bert_f1_excl9, marker="s", linestyle="--", color="tab:orange", label="BERT Macro F1 (Excl. 9)")
plt.plot(samples, fasttext_f1_all, marker="^", color="tab:green", label="FastText Macro F1 (All)")
plt.plot(samples, fasttext_f1_excl6, marker="^", linestyle="--", color="tab:green", label="FastText Macro F1 (Excl. 6)")
plt.xlabel("Number of Training Samples", fontsize=14)
plt.xticks(samples, [str(s) for s in samples])
plt.ylabel("Macro F1-score", fontsize=14)
plt.title("Macro F1 Comparison", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("notebooks/macroF1_comparison_colored.pdf")
plt.show()



