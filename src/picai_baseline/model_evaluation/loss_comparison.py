import matplotlib.pyplot as plt
import json

# --- Replace these with your actual data loading ---
# ε = 3 data (from your example)
with open("/home/zimon/flwr-picai-training/outputs/final_results/DA/epsilon3_DA/averaged_results.json", "r") as f:
    data_eps3 = json.load(f)

# No DP data
with open("/home/zimon/flwr-picai-training/outputs/final_results/DA/no_DP_DA_enabled/averaged_results.json", "r") as f:
    data_no_dp = json.load(f)

# --- Extract loss and rounds ---
rounds_eps3 = [entry["round"] for entry in data_eps3]
loss_eps3 = [entry["central_evaluate_loss"] for entry in data_eps3]

rounds_no_dp = [entry["round"] for entry in data_no_dp]
loss_no_dp = [entry["central_evaluate_loss"] for entry in data_no_dp]

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(rounds_eps3, loss_eps3, label="DP ($\\epsilon=3$)", marker='o')
plt.plot(rounds_no_dp, loss_no_dp, label="No DP", marker='o')

plt.xlabel("Round")
plt.ylabel("Central Evaluation Loss")
plt.title("Loss over Rounds: DP ($\\epsilon=3$) vs No DP")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_dp_vs_nodp.png")
plt.show()
