

# Counting Field on Lattice

基于 Keldysh 形式主义和计数场方法的紧束缚晶格量子输运计算框架。

## 项目简介

本项目实现了一套高效的量子输运计算工具，用于研究晶格系统中的电子输运性质。核心方法基于 Keldysh 路径积分框架下的计数场（Counting Field）技术，并通过 PyTorch 实现自动微分功能，能够计算高阶输运相关量。

## 主要功能

### 1. 输运计算方法

- **计数场导数方法** (`genfunc_cf_deriv_method/`)
  - 使用自动微分计算电流、噪声等输运量的导数
  - 支持高达四阶的导数计算
  - 结合 vmap 方法提升计算效率

- **格林函数逆方法** (`greens_functions_inv_method/`)
  - 直接计算格林函数和输运系数
  - 支持递归格林函数方法
  - 可计算电流密度分布

### 2. 哈密顿模型 (`hamiltonians/`)

- **中心系统** (`Central.py`)
  - `Central`: 标准二维晶格
  - `DisorderedCentral`: 无序晶格系统
  - `CentralBdG`: Bogoliubov-de Gennes 配对形式
  - `TopologicalSurface2D`: 二维拓扑表面态
  - `MZMVortexHamiltonian`: Majorana 零模涡旋结构
  - `ChernTexturedInsulator`: 陈数纹理绝缘体

- **一维模型**
  - `SSHChain`: Su-Schrieffer-Heeger 链
  - `KitaevChain`: Kitaev 链（支持 Majorana 零模）

- **引线** (`Lead.py`)
  - `SpinlessLead`: 无自旋引线
  - `SpinfulLead`: 自旋极化引线
  - `MultiOrbitalLead`: 多轨道引线

### 3. 可视化工具 (`dataplot/`)

- 能带结构与色散关系绘图
- 局部态密度（LDOS）分析
- 电流密度分布可视化
- 输运量随能量变化关系
- 导纳矩阵热图

### 4. 实用工具 (`utils/`)

- 批量张量运算（批量克罗内克积、批量求迹）
- 费米分布函数计算
- 引线消约（Lead Decimation）算法
- 配置参数加载

## 安装依赖

```bash
pip install torch numpy matplotlib
```

项目主要依赖：
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

## 快速开始

### 基本输运计算

```python
import torch
from hamiltonians.Central import Central
from hamiltonians.Lead import SpinfulLead
from greens_functions_inv_method.transport_calculation import calculate_transport_properties

# 定义系统参数
Nx, Ny = 10, 10
t_x = torch.tensor(1.0)
t_y = torch.tensor(1.0)

# 构建中心区域哈密顿量
central = Central(Ny, Nx, t_y, t_x)
H_total = central.H_full

# 定义引线
leads_info = [
    SpinfulLead(mu=torch.tensor(0.0), t_lead=t_x, 
                connection_coordinates=[(0, i) for i in range(Ny)],
                central_Nx=Nx, central_Ny=Ny)
]

# 计算输运性质
E_values = torch.linspace(-2, 2, 100)
temperature = torch.tensor(0.01)
eta = torch.tensor(0.01)

results = calculate_transport_properties(
    E_batch=E_values,
    H_total=H_total,
    leads_info=leads_info,
    temperature=temperature,
    eta=eta
)
```

### 使用计数场方法计算高阶导数

```python
from genfunc_cf_deriv_method.calculations.calculation_cf_autograd import calculation_cf_autograd

# 计算输运量的导数（电流噪声等）
results = calculation_cf_autograd(
    H_BdG=H_BdG,
    E_batch=E_values,
    eta=0.01,
    leads_info=leads_info,
    max_derivative_order=4
)
```

## 项目结构

```
.
├── genfunc_cf_deriv_method/    # 计数场导数方法
│   ├── calculations/           # 自动微分计算
│   └── workflow/              # 格林函数逆构建流程
├── greens_functions_inv_method/ # 格林函数逆方法
│   ├── direct_calculation.py  # 直接计算
│   ├── transport_calculation.py # 输运计算
│   └── total_self_energy.py   # 自能计算
├── hamiltonians/              # 哈密顿模型
│   ├── Central.py             # 中心系统
│   └── Lead.py                # 引线模型
├── dataplot/                  # 数据可视化
│   ├── dispersion_plot.py     # 能带绘图
│   ├── ldos_plot.py           # LDOS绘图
│   ├── current_density_plot.py # 电流密度绘图
│   └── transport_plot.py      # 输运性质绘图
├── utils/                     # 工具函数
│   ├── batch/                 # 批量运算
│   └── physics/               # 物理工具
└── doc/                       # 文档笔记
```

## 应用领域

- 量子霍尔效应研究
- 拓扑绝缘体输运性质
- Majorana 零模探测
- SSH 链噪声分析
- Kitaev 链中的量子输运
- 无序系统中的局域化现象

## 文档

更多理论背景和详细使用方法请参考：

- `doc/note/lattice_generating_slides.md` - 计数场方法理论介绍


## 许可证

本项目仅供研究使用。