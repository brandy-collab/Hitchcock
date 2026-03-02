import streamlit as st
import numpy as np
import pandas as pd

# ==============================================================
#  TRANSPORTE CON COTAS SUPERIORES (0 <= x_ij <= u_ij)
#
#  Criterio de optimalidad (variables NO básicas):
#    - Si está en cota inferior (x_ij = 0):    r_ij >= 0
#    - Si está en cota superior (x_ij = u_ij): r_ij <= 0
#
#  Criterio de cambio de base (variable entrante):
#    - Entra la no básica que más lo viola:
#        min{r_ij : x_ij=0,   r_ij<0}  (más negativo en cota inferior)
#        max{r_ij : x_ij=u_ij, r_ij>0} (más positivo en cota superior)
#
#  Solución inicial:
#    - Se construye una solución factible con heurísticas tipo transporte,
#      respetando u_ij.
#    - Si por cotas quedan ofertas/demandas sin satisfacer, se añaden
#      un origen ficticio (m+1) y un destino ficticio (n+1):
#         a_{m+1} = sum(demandas no satisfechas)
#         b_{n+1} = sum(ofertas no satisfechas)
#      con capacidad ilimitada y coste grande (Big-M), salvo (m+1,n+1)
#      con coste 0.
#
#  Método de optimización:
#    - Simplex de transporte con base tipo árbol (network simplex)
#      y pivotes con límites por cotas inferiores y superiores.
# ==============================================================

TOL = 1e-9


def _clip_to_bounds(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Asegura numéricamente 0 <= x <= U (U puede ser inf)."""
    x = np.where(np.abs(x) < 1e-12, 0.0, x)
    x = np.maximum(x, 0.0)
    fin = np.isfinite(U)
    x = np.where(fin, np.minimum(x, U), x)
    x = np.where(fin & (np.abs(U - x) < 1e-12), U, x)
    return x


def coste_total(x: np.ndarray, C: np.ndarray) -> float:
    return float(np.sum(x * C))

def coste_desglosado(x: np.ndarray, C: np.ndarray, m_orig: int, n_orig: int):
    """Devuelve (z_total, z_original, z_ficticio) separando por celdas originales vs ficticias."""
    m_orig = int(m_orig)
    n_orig = int(n_orig)
    z_orig = float(np.sum(x[:m_orig, :n_orig] * C[:m_orig, :n_orig]))
    z_total = float(np.sum(x * C))
    z_fict = z_total - z_orig
    return z_total, z_orig, z_fict


def fmt_z(z_orig: float, z_fict: float) -> str:
    """Formatea z = z_original + z_ficticio SIN decimales (redondeo al entero más cercano)."""
    zo = int(round(float(z_orig)))
    zf = int(round(float(z_fict)))
    if zf >= 0:
        return f"z = {zo} + {zf}"
    else:
        return f"z = {zo} - {abs(zf)}"




def estado_no_basico(x: np.ndarray, U: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Estado por celda: 'L' (0), 'U' (u), 'F' (interior)."""
    stt = np.full(x.shape, 'F', dtype=object)
    stt[np.abs(x) <= tol] = 'L'
    fin = np.isfinite(U)
    stt[fin & (np.abs(U - x) <= tol)] = 'U'
    return stt


# ---------------- Balanceo total oferta/demanda ----------------


def balancear_problema(S: np.ndarray, D: np.ndarray, C: np.ndarray, U: np.ndarray):
    """Balancea sum(S)=sum(D) añadiendo fila/columna ficticia (coste 0, cap. inf)."""
    S = S.astype(float)
    D = D.astype(float)
    C = C.astype(float)
    U = U.astype(float)

    if np.sum(S) > np.sum(D) + 1e-12:
        D = np.append(D, np.sum(S) - np.sum(D))
        C = np.hstack([C, np.zeros((len(S), 1))])
        U = np.hstack([U, np.full((len(S), 1), np.inf)])
    elif np.sum(D) > np.sum(S) + 1e-12:
        S = np.append(S, np.sum(D) - np.sum(S))
        C = np.vstack([C, np.zeros((1, len(D)))])
        U = np.vstack([U, np.full((1, len(D)), np.inf)])

    return S, D, C, U


# ---------------- Soluciones iniciales respetando u_ij ----------------


def esquina_noroeste_cap(S, D, C, U):
    m, n = len(S), len(D)
    x = np.zeros((m, n))
    s = S.copy()
    d = D.copy()
    cap = U.copy()

    i, j = 0, 0
    while i < m and j < n:
        if s[i] <= TOL:
            i += 1
            continue
        if d[j] <= TOL:
            j += 1
            continue
        if cap[i, j] <= TOL:
            # celda bloqueada
            if j + 1 < n:
                j += 1
            else:
                i += 1
                j = 0
            continue

        val = min(s[i], d[j], cap[i, j])
        x[i, j] += val
        s[i] -= val
        d[j] -= val
        cap[i, j] -= val

        # avance: si se agota oferta o demanda como siempre
        if s[i] <= TOL and d[j] <= TOL:
            i += 1
            j += 1
        elif s[i] <= TOL:
            i += 1
        elif d[j] <= TOL:
            j += 1
        elif cap[i, j] <= TOL:
            # si solo se agota capacidad, avanzamos a la derecha
            j += 1

    return x, s, d


def minimo_costo_fila_cap(S, D, C, U):
    m, n = len(S), len(D)
    x = np.zeros((m, n))
    s = S.copy()
    d = D.copy()
    cap = U.copy()

    for i in range(m):
        while s[i] > TOL:
            disp = [j for j in range(n) if d[j] > TOL and cap[i, j] > TOL]
            if not disp:
                break
            j = disp[int(np.argmin(C[i, disp]))]
            val = min(s[i], d[j], cap[i, j])
            x[i, j] += val
            s[i] -= val
            d[j] -= val
            cap[i, j] -= val

    return x, s, d


def aproximacion_vogel_cap(S, D, C, U):
    m, n = len(S), len(D)
    x = np.zeros((m, n))
    s = S.copy()
    d = D.copy()
    cap = U.copy()

    filas = set(i for i in range(m) if s[i] > TOL)
    cols = set(j for j in range(n) if d[j] > TOL)

    def costos_fila(i):
        return [C[i, j] for j in cols if d[j] > TOL and cap[i, j] > TOL]

    def costos_col(j):
        return [C[i, j] for i in filas if s[i] > TOL and cap[i, j] > TOL]

    while filas and cols:
        penal = []
        for i in list(filas):
            cs = sorted(costos_fila(i))
            if len(cs) == 0:
                filas.remove(i)
                continue
            pen = cs[1] - cs[0] if len(cs) > 1 else cs[0]
            penal.append((pen, 'f', i))
        for j in list(cols):
            cs = sorted(costos_col(j))
            if len(cs) == 0:
                cols.remove(j)
                continue
            pen = cs[1] - cs[0] if len(cs) > 1 else cs[0]
            penal.append((pen, 'c', j))

        if not penal:
            break

        _, tipo, idx = max(penal, key=lambda t: t[0])
        if tipo == 'f':
            i = idx
            cand = [j for j in cols if d[j] > TOL and cap[i, j] > TOL]
            if not cand:
                filas.discard(i)
                continue
            j = cand[int(np.argmin([C[i, jj] for jj in cand]))]
        else:
            j = idx
            cand = [i for i in filas if s[i] > TOL and cap[i, j] > TOL]
            if not cand:
                cols.discard(j)
                continue
            i = cand[int(np.argmin([C[ii, j] for ii in cand]))]

        val = min(s[i], d[j], cap[i, j])
        x[i, j] += val
        s[i] -= val
        d[j] -= val
        cap[i, j] -= val

        if s[i] <= TOL:
            filas.discard(i)
        if d[j] <= TOL:
            cols.discard(j)

    return x, s, d


def añadir_ficticios_por_desequilibrio(S, D, C, U, x, s_res, d_res, bigM=10_000.0):
    """Si por cotas quedan ofertas/demandas sin satisfacer, añade (m+1) y (n+1) con Big-M."""
    unmet_s = float(np.sum(s_res[s_res > TOL]))
    unmet_d = float(np.sum(d_res[d_res > TOL]))
    leftover = max(unmet_s, unmet_d)
    if leftover <= 1e-8:
        return S, D, C, U, x

    # En teoría unmet_s == unmet_d; usamos max por robustez
    unmet_s = leftover
    unmet_d = leftover

    m, n = C.shape
    S2 = np.append(S, unmet_d)  # origen ficticio
    D2 = np.append(D, unmet_s)  # destino ficticio

    C2 = np.zeros((m + 1, n + 1))
    C2[:m, :n] = C
    C2[:m, n] = bigM
    C2[m, :n] = bigM
    C2[m, n] = 0.0

    U2 = np.full((m + 1, n + 1), np.inf)
    U2[:m, :n] = U

    x2 = np.zeros((m + 1, n + 1))
    x2[:m, :n] = x

    # Oferta no satisfecha -> destino ficticio
    for i in range(m):
        if s_res[i] > TOL:
            x2[i, n] = s_res[i]

    # Demanda no satisfecha <- origen ficticio
    for j in range(n):
        if d_res[j] > TOL:
            x2[m, j] = d_res[j]

    x2 = _clip_to_bounds(x2, U2)
    return S2, D2, C2, U2, x2


# ---------------- Base tipo árbol y ciclos ----------------


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def construir_base_arbol(x, U, tol=1e-9):
    """Construye base (m+n-1 celdas) que forme un árbol sobre nodos origen/destino."""
    m, n = x.shape
    N = m + n
    stt = estado_no_basico(x, U, tol)

    # Variables interiores (no en cotas) con valor >0: deben ser básicas
    interiores = [(i, j) for i in range(m) for j in range(n) if stt[i, j] == 'F' and x[i, j] > tol]

    # Caso raro: demasiadas interiores. Se toma un subconjunto sin ciclos.
    if len(interiores) > m + n - 1:
        uf = UnionFind(N)
        base = []
        for (i, j) in sorted(interiores, key=lambda t: float(x[t[0], t[1]]), reverse=True):
            if len(base) >= m + n - 1:
                break
            if uf.union(i, m + j):
                base.append((i, j))
        # completar si falta
        for i in range(m):
            for j in range(n):
                if len(base) >= m + n - 1:
                    break
                if (i, j) in base or U[i, j] <= tol:
                    continue
                if uf.union(i, m + j):
                    base.append((i, j))
        return base

    uf = UnionFind(N)
    base = []
    for (i, j) in interiores:
        if uf.union(i, m + j):
            base.append((i, j))

    # Completar con degeneradas evitando ciclos
    cand_L = [(i, j) for i in range(m) for j in range(n) if (i, j) not in base and stt[i, j] == 'L' and U[i, j] > tol]
    cand_U = [(i, j) for i in range(m) for j in range(n) if (i, j) not in base and stt[i, j] == 'U' and U[i, j] > tol]
    cand_any = [(i, j) for i in range(m) for j in range(n) if (i, j) not in base and U[i, j] > tol]

    for pool in (cand_L, cand_U, cand_any):
        for (i, j) in pool:
            if len(base) >= m + n - 1:
                break
            if uf.union(i, m + j):
                base.append((i, j))
        if len(base) >= m + n - 1:
            break

    return base


def calcular_duales_desde_base(base, C):
    """Potenciales u (filas) y v (columnas) con u_i + v_j = c_ij en la base."""
    m, n = C.shape
    u = {0: 0.0}
    v = {}

    progreso = True
    while progreso and (len(u) < m or len(v) < n):
        progreso = False
        for (i, j) in base:
            if i in u and j not in v:
                v[j] = C[i, j] - u[i]
                progreso = True
            elif j in v and i not in u:
                u[i] = C[i, j] - v[j]
                progreso = True

    u_arr = np.array([u.get(i, 0.0) for i in range(m)], dtype=float)
    v_arr = np.array([v.get(j, 0.0) for j in range(n)], dtype=float)
    return u_arr, v_arr


def encontrar_ciclo_por_arbol(base, entering, m, n):
    """Ciclo único al añadir entering a una base-árbol: [entering] + camino_en_arbol."""
    N = m + n
    adj = [[] for _ in range(N)]
    for (i, j) in base:
        a, b = i, m + j
        adj[a].append((b, (i, j)))
        adj[b].append((a, (i, j)))

    start = entering[0]
    goal = m + entering[1]

    from collections import deque

    q = deque([start])
    prev = {start: None}
    prev_edge = {}

    while q:
        node = q.popleft()
        if node == goal:
            break
        for nb, cell in adj[node]:
            if nb not in prev:
                prev[nb] = node
                prev_edge[nb] = cell
                q.append(nb)

    if goal not in prev:
        return None

    path_cells = []
    cur = goal
    while cur != start:
        path_cells.append(prev_edge[cur])
        cur = prev[cur]
    path_cells.reverse()

    return [entering] + path_cells


# ---------------- Simplex con cotas ----------------


def es_optimo(base_set, x, U, r, tol=1e-9):
    """Comprueba criterio de optimalidad para no básicas (en 0 o en u)."""
    m, n = x.shape
    viol = []
    for i in range(m):
        for j in range(n):
            if (i, j) in base_set or U[i, j] <= tol:
                continue
            if abs(x[i, j]) <= tol:
                if r[i, j] < -tol:
                    viol.append((i, j, float(r[i, j]), 'L'))
            else:
                if np.isfinite(U[i, j]) and abs(U[i, j] - x[i, j]) <= tol:
                    if r[i, j] > tol:
                        viol.append((i, j, float(r[i, j]), 'U'))
                else:
                    # no básica interior (no debería pasar): marcamos si r significativo
                    if abs(r[i, j]) > 10 * tol:
                        viol.append((i, j, float(r[i, j]), 'F'))
    return (len(viol) == 0), viol


def seleccionar_entrante(base_set, x, U, r, tol=1e-9):
    """Selecciona entrante según el criterio del enunciado. Retorna (celda, dir)."""
    m, n = x.shape
    best_L = None  # (r, (i,j)) más negativo en L
    best_U = None  # (r, (i,j)) más positivo en U

    for i in range(m):
        for j in range(n):
            if (i, j) in base_set or U[i, j] <= tol:
                continue
            if abs(x[i, j]) <= tol:
                if r[i, j] < -tol:
                    if best_L is None or r[i, j] < best_L[0]:
                        best_L = (float(r[i, j]), (i, j))
            else:
                if np.isfinite(U[i, j]) and abs(U[i, j] - x[i, j]) <= tol:
                    if r[i, j] > tol:
                        if best_U is None or r[i, j] > best_U[0]:
                            best_U = (float(r[i, j]), (i, j))

    if best_L is None and best_U is None:
        return None
    if best_L is None:
        return best_U[1], -1
    if best_U is None:
        return best_L[1], +1

    # Escoge la más violadora (magnitud)
    if best_U[0] > -best_L[0]:
        return best_U[1], -1
    return best_L[1], +1


def pivot_cotas(base, x, U, entering, entering_dir):
    """Pivote con cotas en un ciclo del árbol."""
    m, n = x.shape
    ciclo = encontrar_ciclo_por_arbol(base, entering, m, n)
    if ciclo is None:
        return None

    # signos alternantes: empieza con entering_dir
    signos = []
    sgn = entering_dir
    for _ in ciclo:
        signos.append(sgn)
        sgn *= -1

    # límites
    limites = []
    for (cell, sgn) in zip(ciclo, signos):
        i, j = cell
        if sgn == +1:
            limites.append((U[i, j] - x[i, j]) if np.isfinite(U[i, j]) else np.inf)
        else:
            limites.append(x[i, j])

    theta = float(np.min(limites))
    if theta < 0:
        theta = 0.0

    idxs = [k for k, lim in enumerate(limites) if abs(lim - theta) <= 1e-10]
    # evita que salga el entrante si hay empate
    leave_idx = None
    for k in idxs:
        if ciclo[k] != entering:
            leave_idx = k
            break
    if leave_idx is None:
        leave_idx = idxs[0]

    leaving = ciclo[leave_idx]

    # actualiza x
    for (cell, sgn) in zip(ciclo, signos):
        i, j = cell
        x[i, j] += sgn * theta

    x = _clip_to_bounds(x, U)

    # actualiza base
    base2 = list(base)
    if entering not in base2:
        base2.append(entering)
    if leaving in base2 and leaving != entering:
        base2.remove(leaving)

    # tamaño objetivo
    target = m + n - 1
    if len(base2) != target:
        # reconstruye una base válida (degeneración)
        base2 = construir_base_arbol(x, U)[:target]

    return base2, x, theta, ciclo, signos, leaving


# ---------------- Estilos ----------------


def resaltar_base(df, celdas_base):
    estilo = pd.DataFrame('', index=df.index, columns=df.columns)
    for r, c in celdas_base:
        if r < estilo.shape[0] and c < estilo.shape[1]:
            estilo.iloc[r, c] = 'background-color: yellow; color: black; font-weight: bold'
    return estilo


def resaltar_cotas(df, x, U, base_set):
    estilo = pd.DataFrame('', index=df.index, columns=df.columns)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (i, j) in base_set:
                continue
            if np.isfinite(U[i, j]) and abs(U[i, j] - x[i, j]) <= 1e-9 and U[i, j] > 1e-9:
                estilo.iloc[i, j] = 'background-color: #d6f5d6; color: black'
    return estilo


# ---------------- UI Streamlit ----------------

st.set_page_config(page_title="Transporte con Cotas", layout="wide")
st.title("🚛 Optimizador de Transporte (con cotas superiores)")
st.info(
    "**Amarillo** = base (variables básicas). **Verde claro** = no básicas en cota superior (x=u).\n"
    "Criterio: no básicas en 0 requieren r≥0; no básicas en u requieren r≤0."
)

with st.sidebar:
    st.header("1. Ajustes")
    metodo = st.selectbox(
        "Método de solución inicial",
        ["Esquina Noroeste", "Costo Mínimo por Fila", "Aproximación de Vogel"],
    )
    max_iter = st.number_input("Máx. iteraciones", min_value=1, max_value=300, value=40)
    bigM = st.number_input("Coste grande (arcos ficticios)", min_value=100.0, value=10000.0, step=100.0)
    st.divider()
    st.header("2. Dimensiones")
    n_or = st.number_input("Orígenes", min_value=2, max_value=12, value=3)
    n_de = st.number_input("Destinos", min_value=2, max_value=12, value=4)

# Defaults
if n_or == 3 and n_de == 4:
    oferta_def, demanda_def = [20, 30, 25], [10, 25, 20, 20]
    costos_def = [[8, 6, 10, 9], [9, 7, 4, 2], [3, 4, 2, 5]]
    caps_def = np.full((n_or, n_de), 9999.0)
else:
    oferta_def, demanda_def = [10] * n_or, [10] * n_de
    costos_def = np.zeros((n_or, n_de))
    caps_def = np.full((n_or, n_de), 9999.0)

cols = st.columns([1, 1, 4])
with cols[0]:
    of_df = st.data_editor(pd.DataFrame(oferta_def, columns=["Oferta"], index=[f"O{i+1}" for i in range(n_or)]))
with cols[1]:
    de_df = st.data_editor(pd.DataFrame(demanda_def, columns=["Demanda"], index=[f"D{j+1}" for j in range(n_de)]))

with cols[2]:
    tabC, tabU = st.tabs(["Costes c_ij", "Cotas u_ij (capacidad máxima)"])
    with tabC:
        c_df = st.data_editor(
            pd.DataFrame(costos_def, columns=[f"D{j+1}" for j in range(n_de)], index=[f"O{i+1}" for i in range(n_or)])
        )
    with tabU:
        st.caption("Usa un número grande (p.ej. 9999) para representar capacidad 'ilimitada'.")
        u_df = st.data_editor(
            pd.DataFrame(caps_def, columns=[f"D{j+1}" for j in range(n_de)], index=[f"O{i+1}" for i in range(n_or)])
        )


if st.button("🚀 Resolver problema"):
    S = of_df["Oferta"].to_numpy(dtype=float)
    D = de_df["Demanda"].to_numpy(dtype=float)
    C = c_df.to_numpy(dtype=float)
    U = u_df.to_numpy(dtype=float)
    # Dimensiones del problema original (antes de añadir ficticios)
    m_orig = len(S)
    n_orig = len(D)


    # u<=0 => arco inexistente
    U = np.where(U <= 0, 0.0, U)

    # 1) balanceo por total
    S_b, D_b, C_b, U_b = balancear_problema(S, D, C, U)

    lbl_r = [f"O{i+1}" for i in range(len(S))] + (["O_Ficticio(balanceo)"] if len(S_b) > len(S) else [])
    lbl_c = [f"D{j+1}" for j in range(len(D))] + (["D_Ficticio(balanceo)"] if len(D_b) > len(D) else [])

    # 2) inicial con capacidades
    if metodo == "Esquina Noroeste":
        x0, s_res, d_res = esquina_noroeste_cap(S_b, D_b, C_b, U_b)
    elif metodo == "Costo Mínimo por Fila":
        x0, s_res, d_res = minimo_costo_fila_cap(S_b, D_b, C_b, U_b)
    else:
        x0, s_res, d_res = aproximacion_vogel_cap(S_b, D_b, C_b, U_b)

    # 3) ficticios por cotas si quedan restos
    S2, D2, C2, U2, x = añadir_ficticios_por_desequilibrio(S_b, D_b, C_b, U_b, x0, s_res, d_res, bigM=float(bigM))
    if C2.shape[0] == len(lbl_r) + 1:
        lbl_r = lbl_r + ["O_Ficticio(cotas)"]
    if C2.shape[1] == len(lbl_c) + 1:
        lbl_c = lbl_c + ["D_Ficticio(cotas)"]

    x = _clip_to_bounds(x, U2)

    # 4) base inicial (árbol)
    base = construir_base_arbol(x, U2)
    base_set = set(base)

    st.subheader(f"🏁 Solución inicial ({metodo})")
    df_init = pd.DataFrame(np.round(x, 6), index=lbl_r, columns=lbl_c)
    st.dataframe(
        df_init.style.apply(resaltar_base, celdas_base=base, axis=None)
        .apply(resaltar_cotas, x=x, U=U2, base_set=base_set, axis=None)
    )
    z_total, z_orig, z_fict = coste_desglosado(x, C2, m_orig, n_orig)
    st.write(f"**Costo inicial:** {fmt_z(z_orig, z_fict)}")

    # 5) iteraciones
    for it in range(1, int(max_iter) + 1):
        base_set = set(base)
        u_pot, v_pot = calcular_duales_desde_base(base, C2)
        r = C2 - u_pot.reshape(-1, 1) - v_pot.reshape(1, -1)

        ok, viol = es_optimo(base_set, x, U2, r)

        z_total, z_orig, z_fict = coste_desglosado(x, C2, m_orig, n_orig)
        with st.expander(f"🔄 Iteración {it} | {fmt_z(z_orig, z_fict)}"):
            st.write(f"**Función objetivo:** {fmt_z(z_orig, z_fict)}")
            df_iter = pd.DataFrame(np.round(x, 6), index=lbl_r, columns=lbl_c)
            st.write("**Asignación actual:**")
            st.dataframe(
                df_iter.style.apply(resaltar_base, celdas_base=base, axis=None)
                .apply(resaltar_cotas, x=x, U=U2, base_set=base_set, axis=None)
            )

            col_u, col_v = st.columns(2)
            with col_u:
                st.dataframe(pd.DataFrame(np.round(u_pot, 6), index=lbl_r, columns=["u"]))
            with col_v:
                st.dataframe(pd.DataFrame(np.round(v_pot, 6), index=lbl_c, columns=["v"]).T)

            st.write("**Costos relativos r_ij:**")
            st.dataframe(pd.DataFrame(np.round(r, 6), index=lbl_r, columns=lbl_c))

            if not ok:
                st.write("**Violaciones (no básicas):**")
                st.dataframe(
                    pd.DataFrame(
                        [{"i": lbl_r[i], "j": lbl_c[j], "r_ij": rij, "cota": t} for (i, j, rij, t) in viol]
                    )
                )

        if ok:
            z_total, z_orig, z_fict = coste_desglosado(x, C2, m_orig, n_orig)
            st.success(f"🏆 ¡Óptimo alcanzado! {fmt_z(z_orig, z_fict)}")
            df_final = pd.DataFrame(np.round(x, 6), index=lbl_r, columns=lbl_c)
            st.dataframe(
                df_final.style.apply(resaltar_base, celdas_base=base, axis=None)
                .apply(resaltar_cotas, x=x, U=U2, base_set=set(base), axis=None)
            )
            break

        sel = seleccionar_entrante(base_set, x, U2, r)
        if sel is None:
            st.warning("No se encontró variable entrante según el criterio (posible degeneración numérica).")
            break

        entering, entering_dir = sel
        piv = pivot_cotas(base, x.copy(), U2, entering, entering_dir)
        if piv is None:
            st.error("No se pudo construir el ciclo: base no conectada (revisar cotas u_ij).")
            break

        base, x, theta, ciclo, signos, leaving = piv

    else:
        st.warning(
            f"Se alcanzó el máximo de iteraciones ({max_iter}) sin certificar optimalidad. "
            f"Función objetivo actual: {fmt_z(*coste_desglosado(x, C2, m_orig, n_orig)[1:])}"
        )
