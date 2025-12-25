import torch

x = torch.tensor(4.0, requires_grad=True)

y = x**3 + 2*x

y.backward()

pytorch_grad = x.grad.item()
print(f"PyTorch Gradient: {pytorch_grad}")

#Check
# f'(x) = 3x^2 + 2 | For x=4: 3(16) + 2 = 50
manual_grad = 3 * (4**2) + 2

if pytorch_grad != manual_grad:
    raise ValueError(f"Gradient mismatch! PyTorch: {pytorch_grad}, Manual: {manual_grad}")
else:
    print("Sanity Check Passed: Gradients match exactly (50.0).")