# 핵심 정리와 증명 해설

이 문서는 공개 이론의 핵심 논증을 짧게 감사할 수 있도록 정리한다.
완전한 원고의 모든 보조정리를 복제하지 않고, 구현과 직접 연결되는
명제에 집중한다.

## 정리 P1 — one-vertex relocation 항등식

투어 \(T\)가 \(uv,wa,wb\)를 포함한다고 하자. \(w\)를 제거하고
\(a,b\)를 연결한 뒤 \(uv\)에 \(w\)를 삽입한 투어 \(T'\)는
단순 Hamiltonian cycle을 유지하는 유효한 교환이라고 가정한다.
\(w\notin\{u,v\}\)이며, self-loop나 중복 간선을 만드는 퇴화 교환은
최소화 대상에서 제외한다. 제거와 추가 양쪽에 같은 간선이 나타나면
그 비용 항은 상쇄된다.

\[
E(T')=
E(T)\setminus\{uv,wa,wb\}
\cup\{uw,wv,ab\}.
\]

따라서

\[
\begin{aligned}
c(T')-c(T)
&=c(uw)+c(wv)+c(ab)\\
&\quad-c(uv)-c(wa)-c(wb)\\
&=[c(uw)+c(wv)-c(uv)]\\
&\quad-[c(wa)+c(wb)-c(ab)]\\
&=s_e(w)-\delta_T(w).
\end{aligned}
\]

그러므로 \(M_T(e)=\min_w[s_e(w)-\delta_T(w)]\ge0\)은 이 이동족에
개선이 없다는 것과 동치다. 다른 2-opt/3-opt 또는 전역 이동까지
막는 조건은 아니다.

## 정리 P2 — raw \(\rho\)의 전역 불충분성

raw \(\rho(e)\)는 \(e\)를 2-hop으로 대체하는 비용만 본다. 간선
주변을 멀리 떨어뜨려 \(\rho(e)\)를 임의로 크게 만들면서도, 투어의
다른 부분에서 더 큰 삭제·교환 이득을 만들어 \(e\)가 없는 tour를
더 싸게 하는 계량 반례족을 구성할 수 있다.

따라서 임의의 고정 \(\tau\)에 대해

\[
\rho(e)>\tau\Longrightarrow
e\text{가 전역 최적투어에 포함}
\]

은 일반적으로 거짓이다. 이 명제는 raw \(\rho\)가 특징량으로
무용하다는 뜻이 아니라, 단독 충분조건이 될 수 없다는 뜻이다.

## 정리 P3 — two-crossing cut 표현

Hamiltonian cycle \(T\)가 cut \((A,B)\)를 정확히 두 번 지난다고
하자. 두 교차간선을 제거하면 각 정점의 차수는 영역 내부에서
2이거나, 교차 끝점이면 1이다. cycle이 연결되어 있으므로 \(A\)와
\(B\) 각각에서 하나의 spanning Hamiltonian path가 남는다.

반대로 두 영역의 spanning path와 그 네 끝점을 연결하는 두
교차간선을 합치면 Hamiltonian cycle이 된다. 따라서 가능한
교차 matching \(M\)에 대해 내부 path 최적값을 최소화하면

\[
\operatorname{OPT}_2(A,B)=
\min_M\left[
H_A(\partial_A M)+H_B(\partial_B M)
+\sum_{e\in M}c(e)
\right].
\]

이 증명은 정확히 두 번 교차하는 층에 한정된다. 더 많은 교차를
배제하려면 별도의 하한이 필요하다.

## 정리 P4 — path-cover–closure 동치

\(G_0\)가 Hamiltonian이 아니라고 하자.

먼저 예외간선이 \(q>0\)개인 cycle에서 그 \(q\)개를 제거한다. 남은
기준 간선은 모든 정점을 덮는 \(q\)개의 서로소 path를 이룬다.
따라서

\[
pc(G_0)\le r_{\mathrm{top}}(G_0).
\]

반대로 최소 path cover의 \(k=pc(G_0)\)개 path를 택한다. 완전
ambient graph에서는 이 path들의 끝점을 cyclic하게 연결하는 간선
\(k\)개를 항상 선택할 수 있다. 그 연결 중 일부가 \(E_0\)에 있으면
예외 수는 더 줄 수 있으므로

\[
r_{\mathrm{top}}(G_0)\le pc(G_0).
\]

따라서 등식이 성립한다. \(G_0\)가 Hamiltonian이면
\(r_{\mathrm{top}}=0\)이므로 path cover의 통상 정의값 1과 구분한다.

## 정리 P5 — exact spectrum과 envelope

모든 tour는 유일한 정수 \(q=r_{G_0}(T)\)에 속하므로 tour 전체
집합은 exact layer의 서로소 합이다. 따라서

\[
z^*=\min_q Z_q^{=}.
\]

누적 허용량 \(q\)의 feasible set은 \(q\)가 커질수록 포함관계로
증가하므로

\[
Z_{\le q+1}\le Z_{\le q}.
\]

정의에서

\[
\begin{aligned}
F_0^{(\le q)}+\Psi_0^{(>q)}
&=(Z_0-Z_{\le q})+(Z_{\le q}-z^*)\\
&=Z_0-z^*=F_0.
\end{aligned}
\]

exact layer \(Z_q^{=}\)의 feasible set끼리는 포함관계가 없으므로
그 값은 단조일 필요가 없다.

## 정리 P6 — pair-closure 항등식

\(e=\{s,t\}\notin E_0\)를 유일한 비기준 간선으로 포함하는 cycle을
생각한다. \(e\)를 제거하면 \(E_0\)만 사용하는 spanning Hamiltonian
\(s\)-\(t\) path가 된다. 반대로 그런 path에 \(e\)를 더하면 정확히
하나의 비기준 간선을 가진 cycle이다.

두 연산은 서로 역이므로 \(e\)를 고정한 최저 cycle 비용은

\[
C_e=H_e+c(e)
\]

이며, 모든 비기준 \(e\)를 최소화하면

\[
Z_1^{=}=\min_{e\notin E_0}[H_e+c(e)].
\]

## 정리 P7 — \(\kappa=1\) 임계점

\(c(e)>0\)라 하자.

\[
\begin{aligned}
\kappa_e>1
&\iff \frac{Z_0-H_e}{c(e)}>1\\
&\iff Z_0-H_e>c(e)\\
&\iff Z_0-[H_e+c(e)]>0\\
&\iff G_e>0\\
&\iff C_e<Z_0.
\end{aligned}
\]

이것은 \(q=0\)과 \(e\)를 고정한 \(q=1\) 상태의 비교다. 전역
forcedness는 모든 다른 \(e'\)와 모든 \(q\ge2\) 층을 비교해야 한다.

## 정리 P8 — 안전 gain sandwich

\[
\underline Z_0\le Z_0\le\overline Z_0,\qquad
\underline H_e\le H_e\le\overline H_e
\]

에서 첫 번째 부등식의 왼쪽과 두 번째의 오른쪽을 사용하면

\[
\underline Z_0-\overline H_e-c(e)
\le Z_0-H_e-c(e)=G_e.
\]

반대 끝을 사용하면

\[
G_e\le\overline Z_0-\underline H_e-c(e).
\]

따라서 \(\underline G_e>0\)은 거짓 양성을 만들지 않는 충분조건이다.
\(\underline G_e\le0\)은 exact \(G_e\le0\)을 뜻하지 않는다.

## 정리 P9 — \(q=1\) support 보존

exact \(q=1\) minimizer \(e^*\)에 대해

\[
\underline C_{e^*}
\le C_{e^*}
=Z_1^{=}
\le\overline Z_1
\]

이다. 따라서 \(\underline C_e>\overline Z_1\)인 pair는 exact
support일 수 없으며

\[
\mathcal A
=\{e:\underline C_e\le\overline Z_1\}
\]

는 exact support를 모두 보존한다.

## 정리 P10 — hull 에너지 분해

정의

\[
S_{\mathrm{int}}=z^*-P_H,\qquad
F_0=Z_0-z^*
\]

를 더하면

\[
S_{\mathrm{int}}+F_0
=z^*-P_H+Z_0-z^*
=Z_0-P_H.
\]

이 항등식은 내부점을 방문하는 본질적 비용과 후보그래프 제약의
추가 비용을 분리한다. 두 항의 통계적 관계나 임계분포는 별도
가설이며 항등식만으로 따라오지 않는다.

## 구현 대응

| 논증 | 공개 코드 |
|---|---|
| exact \(q\)-spectrum | `exception_edge/closure_spectrum.py` |
| endpoint path와 \(q=1\) identity | `exception_edge/closure_threshold.py` |
| 1-tree·Held–Karp 안전구간 | `exception_edge/closure_threshold.py` |
| SEC-MILP 제한 cycle | `exception_edge/subtour_milp.py` |
| forced closure witness 검증 | `exception_edge/lin318_threshold_escape.py` |
| atomic lock/release 2/3-opt | `exception_edge/lin318_basin_escape.py` |
