# 🧠 GPT Fine-tuning with Validation Data

This project demonstrates fine-tuning a **GPT model** with both training and validation datasets. It tracks both `train/loss` and `eval/loss` during the training process using **WandB** for visualization.

---

## 📖 Overview

- **Validation Data**:
  The dataset was split into 80% training and 20% validation data. The validation set allows tracking model performance during training.

- **Training and Evaluation**:
  During training, `train/loss` and `eval/loss` values were logged on **WandB**, allowing for detailed monitoring of the model's learning curve.

---

## 🛠️ Tools and Configuration

- **Model**: Facebook OPT-350M (AutoModelForCausalLM)
- **Dataset**: User-therapist dialog data in JSON format
- **Framework**: Hugging Face Transformers, Datasets, WandB
- **Training Environment**: GPU-based training, 10 epochs
- **Batch Size**: 8
- **Epochs**: 10

---

## 📊 Results

- **Train Loss**: Gradual decrease observed during training, indicating learning progress.
- **Eval Loss**: Reduction over epochs, showing the model's ability to generalize without overfitting.

### Example Graphs:
- Train Loss: ![train loss](https://github.com/user-attachments/assets/9d2274b9-5100-4bb2-a8a6-40b4b67cdbe1)
- Eval Loss: ![eval loss](https://github.com/user-attachments/assets/818996e8-9e8d-4350-84bb-ed1b27d66258)

---

## 🌟 Key Takeaways

This project highlights the importance of validation datasets in fine-tuning. By tracking both `train/loss` and `eval/loss`, it ensures a well-generalized model.

---

## 🔗 WandB Logs

- **Train Loss**: [Link to WandB Train Logs](https://api.wandb.ai/links/wrtyu0603-illinois-institute-of-technology/xkig1j60)
- **Eval Loss**: [Link to WandB Eval Logs](https://api.wandb.ai/links/wrtyu0603-illinois-institute-of-technology/oms9fkrb)
