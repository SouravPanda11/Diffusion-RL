# Diffusion Policies for Offline Reinforcement Learning — *Reimplementation*

This repository is a **from-scratch reimplementation** of the paper:

> **Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning**  
> 📄 Paper: https://arxiv.org/abs/2208.06193

The original authors’ implementation can be found here:

> 🔗 **Official GitHub Repository:**  
> https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL

---

The original codebase depends on **D4RL**, which is difficult to install on Windows, relies on deprecated environments, and conflicts with modern Gymnasium/Python setups—so I rebuilt the entire pipeline **without D4RL**.

Instead: ✔ Trained PPO agents (SB3) — ✔ Collected offline datasets — ✔ Mixed them into one buffer — ✔ Trained diffusion models on the custom dataset

This makes the project: fully reproducible — platform-independent — free of MuJoCo/D4RL issues — simple for students/researchers to run.

---
