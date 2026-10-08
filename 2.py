import torch
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 硬體與晶格網格初始化
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 執行設備: {device}")

GRID_SIZE = 64
L = 4.0  
dx = L / GRID_SIZE

x = torch.linspace(-L/2, L/2, GRID_SIZE, device=device)
y = torch.linspace(-L/2, L/2, GRID_SIZE, device=device)
z = torch.linspace(-L/2, L/2, GRID_SIZE, device=device)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
R = torch.sqrt(X**2 + Y**2 + Z**2)

# -----------------------------------------------------------------------------
# 2. 電子的純電磁幾何設定 (極微小點擾動)
# -----------------------------------------------------------------------------
PHI_BASE = -1.0  
DELTA_PHI_e = 0.276  # 微小電磁漣漪振幅
r_e = 0.1           # 縮小為點電荷近似 (只影響最中心的極少數網格)

# 【正確初始化】：先寫好初始狀態，再交給 PyTorch 追蹤梯度
phi_init = torch.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), PHI_BASE, device=device)
core_mask = R < r_e
phi_init[core_mask] = PHI_BASE + DELTA_PHI_e

# 轉化為可優化參數
phi = phi_init.clone().requires_grad_(True)

# -----------------------------------------------------------------------------
# 3. 定義能量泛函 (無黑洞張力防護罩)
# -----------------------------------------------------------------------------
def compute_electron_energy(phi_field):
    dphi_dx = (torch.roll(phi_field, -1, dims=0) - torch.roll(phi_field, 1, dims=0)) / (2 * dx)
    dphi_dy = (torch.roll(phi_field, -1, dims=1) - torch.roll(phi_field, 1, dims=1)) / (2 * dx)
    dphi_dz = (torch.roll(phi_field, -1, dims=2) - torch.roll(phi_field, 1, dims=2)) / (2 * dx)
    
    grad_sq = dphi_dx**2 + dphi_dy**2 + dphi_dz**2
    E_kinetic = 0.5 * grad_sq

    lam = 20.0
    E_potential = lam * ((phi_field - PHI_BASE)**2) * ((phi_field - 1.0)**2)
    
    total_energy_density = E_kinetic + E_potential
    total_energy = torch.sum(total_energy_density) * (dx**3)
    
    return total_energy

# -----------------------------------------------------------------------------
# 4. 執行變分極小化 (修正 PyTorch 梯度 BUG)
# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam([phi], lr=0.001)

num_steps = 3000
energy_history = []
TARGET_ENERGY = 0.0363  # 從質子(66.69) / 1836.15 推導出的標竿數值

print("\n--- 開始 3D 晶格場模擬 (電子微小漣漪) ---")
for step in range(num_steps):
    optimizer.zero_grad()
    
    loss = compute_electron_energy(phi)
    loss.backward()
    
    # 【正確的物理錨點鎖定寫法】：把不需要更新的梯度直接抹除！
    with torch.no_grad():
        phi.grad[0, :, :] = 0
        phi.grad[-1, :, :] = 0
        phi.grad[:, 0, :] = 0
        phi.grad[:, -1, :] = 0
        phi.grad[:, :, 0] = 0
        phi.grad[:, :, -1] = 0
        # 鎖死核心點的梯度，讓它永遠維持在 DELTA_PHI_e，不受 Adam 動量干擾
        phi.grad[core_mask] = 0

    optimizer.step()
    
    energy_history.append(loss.item())

# -----------------------------------------------------------------------------
# 5. 終端機結果輸出與可視化
# -----------------------------------------------------------------------------
final_electron_energy = loss.item()

print("\n" + "="*50)
print(" 🔬 決戰階段：電子幾何能量驗證")
print("="*50)
print(f"🎯 目標預測能量 (來自質子推導): {TARGET_ENERGY:.6f}")
print(f"✅ 程式實際跑出的電子能量:    {final_electron_energy:.6f}")
print("="*50 + "\n")

# 繪圖
phi_final = phi.detach().cpu().numpy()
center_idx = GRID_SIZE // 2
phi_1d = phi_final[:, center_idx, center_idx]
x_axis = np.linspace(-L/2, L/2, GRID_SIZE)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_axis, phi_1d, 'b-', linewidth=2, label='Electron Ripple $\Phi(x)$')
plt.axhline(PHI_BASE, color='r', linestyle='--', label=r'Base VEV $\langle\Phi\rangle = -1$')
plt.title(f"Electron 1D Ripple Profile")
plt.xlabel("Spatial Coordinate x")
plt.ylabel("Scalar Field Value $\Phi$")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(energy_history, 'k-', linewidth=1.5)
plt.axhline(TARGET_ENERGY, color='r', linestyle='--', label=f'Target: {TARGET_ENERGY:.4f}')
plt.title("Electron Energy Convergence")
plt.xlabel("Relaxation Steps")
plt.ylabel("Total Energy E")
plt.grid(True)
plt.legend()

info_text = (
    f"--- Unified Geometric Theory Result ---\n"
    f"Predicted Electron Energy: {TARGET_ENERGY:.5f} units\n"
    f"Simulated Electron Energy: {final_electron_energy:.5f} units"
)
plt.figtext(0.5, 0.02, info_text, wrap=True, horizontalalignment='center', fontsize=11,
            bbox={'facecolor': 'lightgreen', 'alpha': 0.3, 'pad': 5})

plt.subplots_adjust(bottom=0.2)
plt.savefig("electron_simulation_result.png", dpi=300)
plt.show()