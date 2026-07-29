# 예외간선 이론의 통합 정리

## 0. 문서의 지위

이 문서는 연구 과정에서 따로 발전한 네 층을 하나의 체계로 묶는다.

1. 국소 우회비율과 재배치 마진
2. CUT의 two-crossing 인터페이스
3. Hamiltonian closure spectrum
4. \(\kappa=1\) 임계점과 안전 인증

정리로 증명된 명제, 동결된 표본에서 exact하게 계산된 관찰, 아직
검증할 가설을 의도적으로 분리한다. 세부 증명은
[PROOFS_KO.md](PROOFS_KO.md), 실증 상태는
[CLAIM_LEDGER_KO.md](CLAIM_LEDGER_KO.md)에 정리한다.

---

## 1. 문제 설정과 예외간선의 정확한 의미

### 1.1 기준 후보그래프

정점집합 \(V\), 완전 계량 그래프 \(K_V\), 간선비용 \(c:E(K_V)\to
\mathbb R_{\ge0}\)를 둔다. Euclidean TSP에서는 점 좌표에서 거리를
얻으며, TSPLIB `EUC_2D` 실험에서는 정수 반올림 비용을 사용한다.

희소한 기준 후보그래프를

\[
G_0=(V,E_0),\qquad E_0\subseteq E(K_V)
\]

로 둔다. 본 연구의 주된 \(G_0\)는 weak-Delaunay graph다. 퇴화가 없는
경우 하나의 Delaunay triangulation과 같지만, 공원점 등 퇴화가 있으면
가능한 Delaunay 간선의 합집합과 한 번의 수치적 triangulation을
구분해야 한다.

### 1.2 투어와 인스턴스 수준의 예외

Hamiltonian cycle \(T\)의 기준 밖 간선집합과 예외 수를

\[
X_{G_0}(T)=E(T)\setminus E_0,\qquad
r_{G_0}(T)=|X_{G_0}(T)|
\]

로 정의한다. 따라서 “예외”는 간선의 고유 형용사가 아니라
\(G_0\)에 상대적인 관계다. 기준그래프가 바뀌면 같은 간선의 지위도
바뀐다.

전역 최적값을 \(z^*\), 기준그래프 안의 최적 cycle 값을 \(Z_0\)라
한다.

\[
z^*=\min_T c(T),\qquad
Z_0=\min_{T\subseteq E_0}c(T),
\]

기준 cycle이 없으면 \(Z_0=\infty\)로 둔다.

외부 공개에서는 다음 네 개념을 혼용하지 않는다.

- **비기준 간선:** 단순히 \(e\notin E_0\)
- **optimal-support 예외간선:** 어떤 전역 최적투어가 \(e\)를 포함
- **forced 예외간선:** 모든 전역 최적투어가 \(e\)를 포함
- **mandatory-exception 인스턴스:** 모든 전역 최적투어가 적어도 한
  개의 비기준 간선을 사용, 즉 \(Z_0>z^*\)

어떤 간선이 기준 cycle을 개선한다는 사실만으로 그 간선이
optimal-support 또는 forced가 되지는 않는다.

---

## 2. 국소 기하: raw 우회비율과 재배치 마진

### 2.1 raw 우회비율

간선 \(e=\{u,v\}\)의 단순한 2-hop 우회비율을

\[
\rho(e)=
\min_{w\in V\setminus\{u,v\}}
\frac{c(u,w)+c(w,v)}{c(u,v)}
\]

처럼 둘 수 있다. \(\rho(e)\)가 크면 \(e\) 주변에 짧은 2-hop
대체경로가 없다는 뜻이므로 기하학적 이상신호로 유용하다.

하지만 \(\rho\)는 투어에서 \(w\)를 빼낼 때 회수되는 비용을 보지
않는다. 따라서 임의의 고정 \(\rho\) 임계값만으로 전역 최적투어
포함을 보장할 수 없다. 큰 \(\rho\)와 전역 포함 사이에는 무제한
반례족이 존재한다.

### 2.2 정확한 one-vertex relocation 항등식

투어 \(T\)가 \(e=\{u,v\}\)를 포함하고, 다른 정점 \(w\)의 두
이웃이 \(a,b\)라고 하자. \(w\)를 현재 위치에서 제거해 \(a,b\)를
잇고, \(e\)를 끊어 \(u-w-v\)로 삽입하는 이동을 생각한다.
여기서 최소화의 대상은 이 교환 뒤에도 단순 Hamiltonian cycle이
되는 **유효한 relocation**으로 제한한다. 즉 \(w\notin\{u,v\}\)이고,
self-loop나 중복 간선 때문에 cycle이 퇴화하는 경우는 제외한다.
제거·추가 목록에 같은 무방향 간선이 나타나는 인접 사례는 그 항을
서로 상쇄하는 것으로 해석한다.

삽입 초과비용과 삭제 프리미엄을

\[
s_e(w)=c(u,w)+c(w,v)-c(u,v),
\]

\[
\delta_T(w)=c(w,a)+c(w,b)-c(a,b)
\]

로 두면 새 투어 \(T'\)의 비용 변화는 정확히

\[
c(T')-c(T)=s_e(w)-\delta_T(w)
\]

이다. 따라서

\[
M_T(e)=\min_{w\notin\{u,v\}}
\bigl[s_e(w)-\delta_T(w)\bigr]
\]

는 이 제한된 이동족에 대한 정확한 margin이다.

- \(M_T(e)<0\): 적어도 하나의 one-vertex relocation이 \(T\)를 개선
- \(M_T(e)\ge0\): 이 이동족에서는 개선 불가

이는 전역 최적성 조건이 아니라 특정 국소 이동족에 대한 필요충분
조건이다. raw \(\rho\)는 \(s_e(w)\)의 일부만 보며,
\(\delta_T(w)\)를 포함한 \(M_T(e)\)가 더 정확한 국소량이다.

---

## 3. CUT 층: 두 번 건너는 투어의 정확한 에너지

### 3.1 parity

비자명한 분할 \(V=A\dot\cup B\)를 고정한다. 모든 Hamiltonian
cycle은 cut \(\delta(A)\)를 양의 짝수 번 지난다. 가장 단순한
인터페이스는 정확히 두 번 지나는 상태다.

두 교차간선을 제거하면 \(A\) 안의 spanning Hamiltonian path와
\(B\) 안의 spanning Hamiltonian path가 남는다. 각 영역의 두
끝점을 정하는 교차 matching을 \(M\)이라 하자. 영역 \(S\) 안에서
끝점 집합 \(\partial_S M\)을 갖는 최소 Hamiltonian path 비용을
\(H_S(\partial_S M)\)라 하면, 정확히 두 번 교차하는 최적값은

\[
\operatorname{OPT}_2(A,B)
=
\min_M
\left[
H_A(\partial_A M)
+H_B(\partial_B M)
+\sum_{e\in M}c(e)
\right].
\]

이는 근사가 아니라 two-crossing 층의 정확한 표현이다. CUT은
“긴 간선이 보인다”는 기하 묘사를 영역 내부 접근비용과 경계
교차비용의 결합 문제로 바꾼다.

### 3.2 전역 해석이 가능한 조건

two-crossing 식이 전역 최적투어를 설명하려면 모든 4, 6, ...회
교차 상태가 더 비싸다는 별도 인증이 필요하다. \(q\)-path-cover
하한이나 MST–지름 완화를 이용해

\[
\operatorname{LB}_{\ge4}(A,B)
>
\operatorname{OPT}_2(A,B)
\]

를 증명하면 모든 최적투어가 이 cut을 정확히 두 번 지난다는
충분조건이 된다.

이때 최소에너지 matching family는 전역 최적투어의 경계
인터페이스와 정확히 대응한다. 최소 matching이 유일하면 그 두
교차간선의 강제성을 말할 수 있고, 동률이면 개별 간선의 forcedness를
말할 수 없다.

### 3.3 CUT의 범위

CUT은 선택한 분할과 two-crossing 상태에 대해서는 정확하지만,
모든 비-Delaunay 예외간선의 보편적 분류기는 아니다.

- 적절한 cut을 어떻게 찾을지는 별도 문제다.
- 최적투어가 그 cut을 네 번 이상 지날 수 있다.
- 여러 cut의 인증 간선쌍은 서로 호환되지 않을 수 있다.
- Delaunay graph가 Hamiltonian이어도 비용 때문에 비-Delaunay
  간선이 선택될 수 있다.

따라서 CUT 층은 구조적 강제성을 다루는 한 도구이며, 다음의
closure 층이 기준그래프 전체에 대한 보편적 좌표계를 제공한다.

---

## 4. Hamiltonian closure spectrum

### 4.1 위상적 최소 예외 수

비용을 무시하고 가능한 tour 중 최소 예외 수를

\[
r_{\mathrm{top}}(G_0)=\min_T r_{G_0}(T)
\]

로 둔다. \(G_0\)가 Hamiltonian이면 \(r_{\mathrm{top}}=0\)이다.

\(G_0\)가 Hamiltonian이 아닐 때, 모든 정점을 덮는 서로소
spanning path의 최소 개수를 \(pc(G_0)\)라 하면

\[
r_{\mathrm{top}}(G_0)=pc(G_0).
\]

이 동치는 비기준 간선 \(q\)개를 tour에서 제거하면 기준 간선만으로
된 path \(q\)개가 남고, 반대로 path cover의 끝점들을 완전그래프에서
cyclic하게 닫을 수 있다는 데서 나온다.

중요한 결과는 보편적인 양의 예외비율 하한이 없다는 것이다.
기준그래프 자체가 Hamiltonian이면 위상적으로 필요한 예외 수는
0이다. 실제 비-Delaunay 선택의 핵심은 단순 feasibility보다 비용
frustration에 있다.

### 4.2 exact \(q\)-layer

예외 수가 정확히 \(q\)인 최저 tour 값을

\[
Z_q^{=}
=
\min\{c(T):r_{G_0}(T)=q\}
\]

로 정의하고, 불가능하면 \(\infty\)로 둔다. 그러면

\[
z^*=\min_{q\ge0}Z_q^{=}.
\]

최적투어가 나타나는 예외 수의 집합과 최소값을

\[
Q^*=\{q:Z_q^{=}=z^*\},\qquad q^*=\min Q^*
\]

로 둔다.

`exactly q` 층 \(Z_q^{=}\)는 \(q\)에 대해 단조일 필요가 없다.
단조량은 누적 envelope

\[
Z_{\le q}=\min_{0\le j\le q}Z_j^{=}
\]

이며, \(q\)가 증가하면 비증가한다.

### 4.3 frustration 회수 계층

기준그래프 제약으로 생기는 총 비용손실을

\[
F_0=Z_0-z^*
\]

라 하자. 예외간선을 최대 \(q\)개 허용해 회수한 부분과 남은 부분을

\[
F_0^{(\le q)}=Z_0-Z_{\le q},
\qquad
\Psi_0^{(>q)}=Z_{\le q}-z^*
\]

로 두면 정확히

\[
F_0=F_0^{(\le q)}+\Psi_0^{(>q)}
\]

이다. 이 식은 “단일 예외간선이 얼마를 설명하는가”와 “두 개 이상이
함께 작동해야만 회수되는 잔여가 얼마인가”를 분리한다.

---

## 5. \(q=1\) pair closure와 \(\kappa=1\) 임계점

### 5.1 끝점 Hamiltonian path

비기준 간선 \(e=\{s,t\}\notin E_0\)에 대해

\[
H_e
=
\min\{c(P):
P\subseteq E_0,\;
P\text{는 spanning Hamiltonian }s\text{-}t\text{ path}\}
\]

를 둔다. 그런 path가 없으면 \(H_e=\infty\)다.

\(P\)에 \(e\)를 더하면 예외간선이 정확히 하나인 cycle이 되고,
반대로 \(e\) 하나를 가진 cycle에서 \(e\)를 빼면 그런 path가 된다.
따라서

\[
C_e=H_e+c(e)
\]

는 \(e\)를 유일한 비기준 간선으로 사용하는 최저 cycle 비용이고,

\[
Z_1^{=}=\min_{e\notin E_0} C_e
\]

이다.

### 5.2 release, gain, 임계량

\[
R_e=Z_0-H_e,\qquad
G_e=Z_0-H_e-c(e)=Z_0-C_e,
\]

\[
\kappa_e=\frac{R_e}{c(e)}
=\frac{Z_0-H_e}{c(e)}
\]

로 둔다. 그러면

\[
\boxed{
\kappa_e>1
\iff
G_e>0
\iff
C_e<Z_0
}
\]

이다.

해석은 단순하다. 기준 cycle 제약을 \(s,t\)에서 열어 얻는 비용
해방량 \(R_e\)가 그 끝점을 다시 비기준 간선으로 닫는 비용 \(c(e)\)
보다 클 때 \(q=1\) 개선이 발생한다.

\(\kappa=1\)은 유한 문제의 정확한 대수적 경계다. 이를 열역학적
상전이의 증명으로 부르지는 않는다. 물리적 비유는 직관을 주지만,
증명되는 것은 위의 항등식이다.

### 5.3 무엇이 따라오지 않는가

\(\kappa_e>1\)에서 다음은 자동으로 따라오지 않는다.

- \(C_e=z^*\)
- \(e\)가 어떤 전역 최적투어에 포함
- \(e\)가 모든 전역 최적투어에 포함
- \(q=2,3,\ldots\) 층보다 \(q=1\)이 우월
- 현재 국소최적점에서 작은 한 번의 k-opt로 \(C_e\)에 도달

따라서 \(\kappa\)-positive 간선은 “정답 간선”이라기보다
\(q=1\) closure 또는 다른 basin으로 들어가는 구조적 portal
후보다.

---

## 6. exact 계산을 안전한 인증으로 바꾸기

### 6.1 gain 구간

대규모에서는 \(Z_0\)와 모든 \(H_e\)의 exact 계산이 어렵다.
다음 검증 가능한 구간을 둔다.

\[
\underline Z_0\le Z_0\le\overline Z_0,
\qquad
\underline H_e\le H_e\le\overline H_e.
\]

그러면

\[
\underline G_e
=
\underline Z_0-\overline H_e-c(e)
\le G_e
\le
\overline Z_0-\underline H_e-c(e)
=\overline G_e.
\]

마찬가지로

\[
\underline\kappa_e
=\frac{\underline Z_0-\overline H_e}{c(e)}
\le\kappa_e
\le
\frac{\overline Z_0-\underline H_e}{c(e)}
=\overline\kappa_e.
\]

- \(\underline G_e>0\): 안전 양성
- \(\overline G_e\le0\): 안전 음성
- 그 사이: 미확정

하한이 0 이하라는 이유만으로 음성이라 부르면 안 된다.

### 6.2 bound의 출처

cycle 쪽 \(\underline Z_0\)에는 1-tree 또는 Held–Karp potential
하한, path 쪽 \(\underline H_e\)에는 끝점 조건을 반영한 forced
1-tree 하한을 쓸 수 있다. \(\overline Z_0\)는 검증된 feasible
baseline cycle, \(\overline H_e\)는 검증된 feasible baseline
Hamiltonian path가 제공한다.

LIN318에서는 weak-Delaunay 제한 MILP의 유효한 하한과 검증된 tour
상한이 모두 42,231로 일치했다. 각 \(e\)에 대해 LKH가 만든
`weak-Delaunay + e` forced cycle을 독립 재검사하고 \(e\)를 제거해
\(\overline H_e\)를 얻었다. 따라서

\[
\underline G_e
=42{,}231-\overline H_e-c(e)>0
\]

인 13개만 안전 양성으로 보고했다. 이 계산은 exact \(H_e\)나 exact
\(\kappa_e\)를 구한 것이 아니다.

### 6.3 \(q=1\) support 보존 상한

각 pair의 closure 하한과 feasible 상한을

\[
\underline C_e=\underline H_e+c(e),\qquad
\overline C_e=\overline H_e+c(e)
\]

라 하고,

\[
\overline Z_1=\min_e\overline C_e
\]

라 하자. 그러면

\[
\mathcal A
=
\{e\notin E_0:
\underline C_e\le\overline Z_1\}
\]

는 exact \(q=1\) minimizer support를 모두 포함한다. 즉 안전하게
후보를 줄일 수 있지만, \(\mathcal A\)의 각 간선이 최적이라는 뜻은
아니다.

---

## 7. convex hull, 내부점, 인접에너지

### 7.1 convex ground calibration

모든 점이 convex hull 위에 있고 일반 위치에 있으면 hull 순서의
cycle이 Euclidean optimum이며

\[
z^*=P_H
\]

이다. 여기서 \(P_H\)는 hull perimeter다. 내부점이 생기면 투어는
hull edge를 열고 내부 정점을 통과하는 chain을 삽입해야 한다.

### 7.2 내부 strain과 기준그래프 frustration

두 양을 구분한다.

\[
S_{\mathrm{int}}=z^*-P_H
\]

는 내부점을 모두 방문하기 위해 완전그래프 optimum 자체가 부담하는
intrinsic strain이다.

\[
F_0=Z_0-z^*
\]

는 기준 후보그래프 제약 때문에 추가로 생기는 frustration이다.

따라서

\[
Z_0-P_H
=
\underbrace{(z^*-P_H)}_{S_{\mathrm{int}}}
+
\underbrace{(Z_0-z^*)}_{F_0}.
\]

내부점이 많아져 \(S_{\mathrm{int}}\)가 커져도 \(F_0=0\)일 수 있다.
내부 경로의 복잡성과 비-Delaunay 예외 발생을 동일시하면 안 된다.

### 7.3 shell-chain relaxation과 가설

hull gap마다 내부 정점을 배정해 chain 비용을 최소화하는
shell-chain relaxation은 인접 압력의 하한 또는 사전 특징을 제공할
수 있다. 그러나 relaxation gap이 자동으로 \(F_0\)의 exact
상·하한이 되는 것은 아니다.

현재의 사전 가설은 다음과 같다.

- shell pressure의 공간적 집중이 mandatory 또는 \(q=1\) 발생을
  예측할 수 있다.
- strain의 분산·집중 패턴이 \(q^*=1\)과 \(q^*\ge2\)를 구분할 수 있다.

이는 closure 항등식처럼 증명된 명제가 아니라 다음 holdout 실험의
대상이다.

---

## 8. 통합 상태도

기준그래프 \(G_0\)에 대해 다음 상태를 구분할 수 있다.

### 상태 A — 기준 완전성

\[
Z_0=z^*,\qquad F_0=0.
\]

기준그래프 안에 전역 최적투어가 존재한다. 내부점이 있어
\(S_{\mathrm{int}}>0\)이어도 가능하다.

### 상태 B — first-order instability

\[
Z_1^{=}<Z_0.
\]

어떤 \(\kappa_e>1\)이 존재해 한 개 closure edge가 기준 cycle을
개선한다.

### 상태 C — first-order completeness

\[
Z_1^{=}=z^*<Z_0.
\]

\(q=1\) 층만으로 전역 optimum 비용에 도달한다. 어떤 특정 \(e\)의
forcedness는 별도 문제다.

### 상태 D — higher-order onset

\[
Z_1^{=}>z^*,\qquad
\min_{q\ge2}Z_q^{=}=z^*.
\]

두 개 이상의 예외간선 호환성이 필요하다.

### 상태 E — 위상적 deficit

\[
r_{\mathrm{top}}(G_0)>0.
\]

기준그래프 자체가 Hamiltonian이 아니어서 비용과 무관하게 예외가
필요하다.

Stage 2와 3의 exact-small 표본에서는 상태 B/C, 특히 \(q=1\)이
강하게 집중되었다. LIN318의 안전 양성 13개는 상태 B를 보장하는
개별 witness다. 하지만 42,108과 42,029 사이의 잔여 79가 상태 D의
bundle 문제인지, 탐색 연산자 한계인지, 아직 미확정 pair 때문인지는
결론나지 않았다.

---

## 9. 알고리즘적 의미

이 이론은 완전그래프 조합탐색을 즉시 다항시간 기하문제로 바꾸지
않는다. \(H_e\) 자체가 Hamiltonian path 문제이며, 여러 closure
간선의 호환성도 조합적이다. 그러나 탐색의 좌표계를 바꾼다.

1. 기준그래프로 넓은 국소 구조를 설명한다.
2. 비기준 간선을 \(q\)-layer로 분해한다.
3. \(q=1\)은 endpoint path와 간선 하나의 정확한 비교로 환원한다.
4. 1-tree와 feasible witness로 안전한 양성·음성·미확정을 나눈다.
5. 안전 양성을 단순 삽입 정답이 아니라 basin portal 후보로 쓴다.
6. 남은 잔여를 2-edge/3-edge bundle 호환성 문제로 올린다.

즉 난이도를 제거한 것이 아니라, “어디에 조합적 난이도가 남았는가”를
더 작고 검증 가능한 층으로 이동시킨 것이 현재의 성과다.

---

## 10. 다음 이론 과제

1. **bundle closure:** \(q=2,3\)에서 개별 \(\kappa\)의 합이 아닌
   상호작용항을 정의한다.
2. **portal compatibility graph:** 두 closure 간선이 하나의
   Hamiltonian tour에서 공존할 수 있는지와 순서 제약을 표현한다.
3. **CUT–closure 연결:** 유리한 endpoint pair가 어떤 cut의
   two-crossing matching으로 나타나는지 충분조건을 찾는다.
4. **사전 predictor:** exact/forced closure 계산 전에 hull pressure,
   onion depth, corridor, endpoint accessibility로 후보를 순위화한다.
5. **규모·분포 holdout:** 여러 TSPLIB 및 새 좌표분포에서 사전 동결한
   예산과 대조군으로 portal 효과를 평가한다.

이 과제들이 성공해도 NP-hardness가 자동으로 사라지는 것은 아니다.
완전 해결을 주장하려면 모든 필요한 \(q\)-층을 다항 크기의 안전
후보와 다항시간 호환성 판정으로 닫아야 한다. 현재 연구는 그 중
\(q=1\) 층을 가장 명확하게 닫은 단계다.
