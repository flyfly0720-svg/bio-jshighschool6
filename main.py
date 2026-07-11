
"""
YAP/TAZ 기반 지방생성 '비만 스위치' 양쌍안정성(bistability) 시뮬레이터
------------------------------------------------------------------
생물학적 배경:
- 기질 강성/세포 퍼짐 등 역학적 신호는 핵 내 YAP/TAZ 활성을 조절함
  (Dupont et al. 2011, Nature, "Role of YAP/TAZ in mechanotransduction")
- TAZ는 지방생성 핵심 전사인자 PPARγ의 보조억제인자로 작용해
  FABP4·아디포넥틴·GLUT4 등 지질대사 유전자 발현을 억제함
  (Hong et al. 2005; TAZ adipocyte-knockout 연구들)

수학적 틀:
- Gardner, Cantor & Collins (2000) Nature, "Construction of a genetic
  toggle switch in Escherichia coli"의 상호억제 힐함수 토글스위치 구조를
  TAZ-PPARγ 상호억제 회로에 맞게 재설계함.

    du/dt = a1 / (1 + v**n1) - g1*u      (u: 핵 내 활성 TAZ)
    dv/dt = a2 / (1 + u**n2) - g2*v      (v: 활성 PPARγ, 지질대사 스위치)

실행 방법: pip install -r requirements.txt
           streamlit run taz_ppar_toggle_switch_app.py
"""

import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="TAZ-PPARγ 비만 스위치 시뮬레이터", layout="wide")

st.title("🧬 YAP/TAZ 기반 지방생성 '비만 스위치' 시뮬레이터")
st.markdown("""
기질 강성 같은 **역학적 신호 → TAZ 활성 → PPARγ 억제**로 이어지는 상호억제 회로가
중간 상태 없이 두 가지 대사 상태(지방세포 분화 vs 미분화)만을 만들어내는
**양쌍안정성(bistability)**을 가지는지 살펴봅니다.

- **u** : 핵 내 활성 TAZ (기질이 단단할수록, 세포가 넓게 퍼질수록 증가)
- **v** : 활성 PPARγ (지질 합성·저장 유전자 발현을 이끄는 대사 스위치)
""")

# ---------------- 사이드바 ----------------
st.sidebar.header("① 토글스위치 매개변수")
a1 = st.sidebar.slider("a1 (TAZ 최대 생성률)", 0.5, 6.0, 3.0, 0.1)
a2 = st.sidebar.slider("a2 (PPARγ 최대 생성률)", 0.5, 6.0, 3.0, 0.1)
n_hill = st.sidebar.slider("n (힐 계수 = 협동성, TAZ·PPARγ 동일 적용)", 0.5, 4.0, 2.0, 0.1)
g1 = st.sidebar.slider("g1 (TAZ 분해율)", 0.2, 3.0, 1.0, 0.1)
g2 = st.sidebar.slider("g2 (PPARγ 분해율)", 0.2, 3.0, 1.0, 0.1)

st.sidebar.header("② 초기 조건 (기질 강성 대리변수)")
preset = st.sidebar.radio(
    "빠른 설정",
    ["직접 입력", "연한 기질 (지방분화 유도)", "단단한 기질 (지방분화 억제)"]
)
if preset == "연한 기질 (지방분화 유도)":
    u0_default, v0_default = 0.1, 3.0
elif preset == "단단한 기질 (지방분화 억제)":
    u0_default, v0_default = 3.0, 0.1
else:
    u0_default, v0_default = 1.0, 1.0

u0 = st.sidebar.slider("u(0): 초기 TAZ 활성", 0.0, 4.0, float(u0_default), 0.1)
v0 = st.sidebar.slider("v(0): 초기 PPARγ 활성", 0.0, 4.0, float(v0_default), 0.1)
t_end = st.sidebar.slider("시뮬레이션 시간", 10, 100, 50, 5)

# ---------------- 모델 ----------------
def toggle_switch(t, state, a1, a2, n1, n2, g1, g2):
    u, v = state
    dudt = a1 / (1 + v**n1) - g1 * u
    dvdt = a2 / (1 + u**n2) - g2 * v
    return [dudt, dvdt]

t_span = (0, t_end)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(toggle_switch, t_span, [u0, v0],
                 args=(a1, a2, n_hill, n_hill, g1, g2), t_eval=t_eval)

# ---------------- 시간 경과 그래프 ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("시간에 따른 활성 변화")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sol.t, sol.y[0], label="u (핵 내 TAZ)")
    ax1.plot(sol.t, sol.y[1], label="v (활성 PPARγ)")
    ax1.set_xlabel("시간")
    ax1.set_ylabel("활성 농도")
    ax1.legend()
    st.pyplot(fig1)

    final_state = "TAZ 우세 → 지방분화 억제(비만 저항)" if sol.y[0, -1] > sol.y[1, -1] \
        else "PPARγ 우세 → 지방분화 진행(지질 축적)"
    st.info(f"현재 조건에서 최종 상태: **{final_state}**")

with col2:
    st.subheader("위상평면 (u-v) 궤적과 널클라인")
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    v_range = np.linspace(0.01, 4, 300)
    u_nullcline = a1 / (1 + v_range**n_hill) / g1
    ax2.plot(u_nullcline, v_range, "--", color="tab:blue", label="du/dt = 0")

    u_range = np.linspace(0.01, 4, 300)
    v_nullcline = a2 / (1 + u_range**n_hill) / g2
    ax2.plot(u_range, v_nullcline, "--", color="tab:orange", label="dv/dt = 0")

    ax2.plot(sol.y[0], sol.y[1], color="black", label="현재 궤적")
    ax2.scatter([u0], [v0], color="green", zorder=5, label="시작점")
    ax2.scatter([sol.y[0, -1]], [sol.y[1, -1]], color="red", zorder=5, label="도착점")
    ax2.set_xlabel("u (TAZ)")
    ax2.set_ylabel("v (PPARγ)")
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

# ---------------- 초기조건 그리드로 양쌍안정성 지도 ----------------
st.subheader("🗺️ 초기조건에 따른 최종 상태 지도 (양쌍안정성 확인)")
st.caption("여러 초기조건(u0, v0)에서 출발한 궤적이 어느 안정 상태로 수렴하는지 격자 형태로 보여줍니다.")

grid_n = 12
grid_vals = np.linspace(0.05, 4, grid_n)
final_map = np.zeros((grid_n, grid_n))

for i, uu in enumerate(grid_vals):
    for j, vv in enumerate(grid_vals):
        s = solve_ivp(toggle_switch, (0, 60), [uu, vv],
                      args=(a1, a2, n_hill, n_hill, g1, g2), t_eval=[60])
        final_map[j, i] = 1 if s.y[0, -1] > s.y[1, -1] else 0  # 1: TAZ우세, 0: PPARg우세

fig3, ax3 = plt.subplots(figsize=(5, 4.5))
im = ax3.imshow(final_map, origin="lower", extent=[0, 4, 0, 4], cmap="coolwarm", aspect="auto")
ax3.set_xlabel("u(0): 초기 TAZ")
ax3.set_ylabel("v(0): 초기 PPARγ")
ax3.set_title("빨강: TAZ 우세(비지방) / 파랑: PPARγ 우세(지방분화)")
st.pyplot(fig3)

n_taz = int(final_map.sum())
n_ppar = grid_n * grid_n - n_taz
if n_taz > 0 and n_ppar > 0:
    st.success(f"현재 n={n_hill:.1f}에서는 초기조건에 따라 두 가지 최종 상태가 모두 나타나는 "
               f"**양쌍안정성**이 확인됩니다 (TAZ우세 {n_taz}개 / PPARγ우세 {n_ppar}개).")
else:
    st.warning(f"현재 n={n_hill:.1f}에서는 모든 초기조건이 하나의 상태로 수렴하는 "
               "**단일 안정 상태(그레이디드 반응)**입니다. n(힐 계수)을 2 이상으로 올려보세요.")

st.markdown("""
---
**해석 가이드:** n(힐 계수, 협동성)을 1.5 이하로 낮추면 모든 초기조건이 같은 최종 상태로
수렴하는 그레이디드 반응이 되고, n을 2 이상으로 올리면 초기조건(=기질 강성)에 따라
전혀 다른 두 안정 상태로 갈라지는 스위치형 반응이 나타납니다. 이는 TAZ-PPARγ 상호억제
회로가 협동적 결합을 가질 때 비로소 '전부 아니면 전무(all-or-none)' 방식의 지방분화
결정, 즉 비만 스위치로 작동할 수 있음을 보여줍니다.
""")
