import torch
import matplotlib.pyplot as plt



"""

Run the code: 

python src/franka_ai/tests/test_lr_scheduler.py 

"""



# Dummy parameters (to attach optimizer)
params = [torch.nn.Parameter(torch.randn(1))]

# Hyperparameters
learning_rate = 0.0005
lr_warmup_steps = 500
training_steps = 24000

# Optimizer
optimizer = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

# Linear LR warmup
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,  # start at 10% of lr
    total_iters=lr_warmup_steps
)

# Cosine LR after warmup
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=training_steps - lr_warmup_steps,
    eta_min=learning_rate * 0.1
)

# Sequential scheduler
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[lr_warmup_steps]
)

# Track LR
lrs = []
for step in range(training_steps):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plot
plt.figure(figsize=(8,4))
plt.plot(range(training_steps), lrs)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Warmup + Cosine LR schedule")
plt.grid(True)
plt.show()
