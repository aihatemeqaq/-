import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 硬體與晶格網格初始化 (GPU 加速)
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 執行設備: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU 型號: {torch.cuda.get_device_name(0)}")

GRID_SIZE = 64
L = 4.0  
dx = L / GRID_SIZE

x = torch.linspace(-L/2, L/2, GRID_SIZE, device=device)
y = torch.linspace(-L/2, L/2, GRID_SIZE, device=device)
z = torch.linspace(-L/2, L/2, GRID_SIZE, device=device)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
R = torch.sqrt(X**2 + Y**2 + Z**2)

# -----------------------------------------------------------------------------
# 2. 理論參數與場初始條件設定
# -----------------------------------------------------------------------------
PHI_BASE = -1.0  
PHI_PROTON = +1.0 
DELTA_PHI = PHI_PROTON - PHI_BASE 

r_0 = 1.2  
phi = torch.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), PHI_BASE, device=device, requires_grad=True)

with torch.no_grad():
    phi[:] = PHI_BASE + DELTA_PHI * torch.exp(- (R / r_0)**4)

# -----------------------------------------------------------------------------
# 3. 定義幾何拉格朗日量與能量泛函
# -----------------------------------------------------------------------------
def compute_total_energy(phi_field, T_BH):
    dphi_dx = (torch.roll(phi_field, -1, dims=0) - torch.roll(phi_field, 1, dims=0)) / (2 * dx)
    dphi_dy = (torch.roll(phi_field, -1, dims=1) - torch.roll(phi_field, 1, dims=1)) / (2 * dx)
    dphi_dz = (torch.roll(phi_field, -1, dims=2) - torch.roll(phi_field, 1, dims=2)) / (2 * dx)
    
    grad_sq = dphi_dx**2 + dphi_dy**2 + dphi_dz**2
    E_kinetic = 0.5 * grad_sq

    lam = 20.0
    E_potential = lam * ((phi_field - PHI_BASE)**2) * ((phi_field - PHI_PROTON)**2)

    # 邊界薄殼防護罩 (Topological Shell)
    shell_thickness = 0.4
    pocket_mask = torch.exp(-((R - r_0) / shell_thickness)**2)
    E_boundary = - T_BH * pocket_mask * (phi_field - PHI_BASE)**2

    total_energy_density = E_kinetic + E_potential + E_boundary
    total_energy = torch.sum(total_energy_density) * (dx**3)
    
    return total_energy

# -----------------------------------------------------------------------------
# 4. 執行變分極小化
# -----------------------------------------------------------------------------
T_BH = 7.75  # 完美的黃金臨界點
optimizer = torch.optim.Adam([phi], lr=0.001)

num_steps = 10000
energy_history = []

print("\n--- 開始 3D 晶格場弛豫模擬 ---")
for step in range(num_steps):
    optimizer.zero_grad()
    
    with torch.no_grad():
        phi[0, :, :] = PHI_BASE
        phi[-1, :, :] = PHI_BASE
        phi[:, 0, :] = PHI_BASE
        phi[:, -1, :] = PHI_BASE
        phi[:, :, 0] = PHI_BASE
        phi[:, :, -1] = PHI_BASE

    loss = compute_total_energy(phi, T_BH)
    loss.backward()
    optimizer.step()
    
    energy_history.append(loss.item())

# -----------------------------------------------------------------------------
# 5. 質量比例計算與終端機輸出 (在顯示圖片前印出！)
# -----------------------------------------------------------------------------
proton_mass_energy = abs(loss.item())
REAL_MASS_RATIO = 1836.15
predicted_electron_energy = proton_mass_energy / REAL_MASS_RATIO

print("\n" + "="*40)
print(" 階段二：物理量數值對齊分析")
print("="*40)
print(f"📦 質子口袋封印的總張力能量 (E_proton): {proton_mass_energy:.4f}")
print(f"🔬 根據實測質量比 1836.15，預測電子在此晶格空間中的微小擾動能量 (E_electron): {predicted_electron_energy:.6f}")
print("="*40 + "\n")

# -----------------------------------------------------------------------------
# 6. 可視化繪圖 (帶有黃色數據方塊)
# -----------------------------------------------------------------------------
phi_final = phi.detach().cpu().numpy()
center_idx = GRID_SIZE // 2
phi_1d = phi_final[:, center_idx, center_idx]
x_axis = np.linspace(-L/2, L/2, GRID_SIZE)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_axis, phi_1d, 'b-', linewidth=2, label='Scalar Field $\Phi(x)$')
plt.axhline(PHI_BASE, color='r', linestyle='--', label='Base VEV $\langle\Phi\\rangle = -1$')
plt.axhline(PHI_PROTON, color='g', linestyle='--', label='Proton State $\Phi_p = +1$')
plt.title(f"Proton Soliton 1D Profile (T_BH = {T_BH})")
plt.xlabel("Spatial Coordinate x")
plt.ylabel("Scalar Field Value $\Phi$")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(energy_history, 'k-', linewidth=1.5)
plt.title("Total Energy Convergence ($\delta E / \delta \Phi \\to 0$)")
plt.xlabel("Relaxation Steps")
plt.ylabel("Total Energy E")
plt.grid(True)

# 加入圖表下方的黃色數據方塊
info_text = (
    f"--- Physical Mass Alignment ---\n"
    f"Proton Soliton Energy (Mass): {proton_mass_energy:.2f} units\n"
    f"Real Proton/Electron Mass Ratio: 1836.15\n"
    f"Predicted Electron Background Fluctuation: {predicted_electron_energy:.4f} units"
)
plt.figtext(0.5, 0.02, info_text, wrap=True, horizontalalignment='center', fontsize=11,
            bbox={'facecolor': 'yellow', 'alpha': 0.3, 'pad': 5})

plt.subplots_adjust(bottom=0.2)
plt.savefig("soliton_simulation_with_mass.png", dpi=300)
plt.show()  # 這行會暫停程式直到你關閉圖片，所以剛才把 print 移到這行上面了！