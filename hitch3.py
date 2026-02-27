import streamlit as st
import numpy as np
import pandas as pd

# --- 1. ALGORITMOS PRINCIPALES ---

def balancear_problema(S, D, C):
    S, D = S.astype(float), D.astype(float)
    if sum(S) > sum(D):
        D = np.append(D, sum(S) - sum(D))
        C = np.hstack((C, np.zeros((len(S), 1))))
    elif sum(D) > sum(S):
        S = np.append(S, sum(D) - sum(S))
        C = np.vstack((C, np.zeros((1, len(D)))))
    return S, D, C

def asegurar_m_n_1(x, S, D):
    """
    Garantiza que existan exactamente m + n - 1 variables b?sicas.
    Si faltan, a?ade ceros en celdas que no formen ciclos.
    """
    m, n = x.shape
    objetivo = m + n - 1
    base = obtener_base(x)
    
    if len(base) < objetivo:
        # Intentamos a?adir celdas vac?as como 'b?sicas' con valor 0
        for i in range(m):
            for j in range(n):
                if (i, j) not in base:
                    # Probamos si a?adir esta celda crea un ciclo
                    if obtener_ciclo(base + [(i, j)], (i, j)) is None:
                        base.append((i, j))
                        x[i, j] = 1e-10 # Un cero 't?cnico' muy peque?o
                        if len(base) == objetivo:
                            return x
    return x

def esquina_noroeste(S, D):
    x = np.zeros((len(S), len(D)))
    s_copy, d_copy = S.copy(), D.copy()
    i, j = 0, 0
    while i < len(S) and j < len(D):
        val = min(s_copy[i], d_copy[j])
        x[i, j] = val
        s_copy[i] -= val
        d_copy[j] -= val
        if s_copy[i] < 1e-9: i += 1
        elif d_copy[j] < 1e-9: j += 1
    return asegurar_m_n_1(x, S, D)

def minimo_costo_fila(S, D, C):
    x = np.zeros((len(S), len(D)))
    s_copy, d_copy = S.copy(), D.copy()
    for i in range(len(S)):
        while s_copy[i] > 1e-9:
            disp_j = [j for j in range(len(D)) if d_copy[j] > 1e-9]
            if not disp_j: break
            mejor_j = disp_j[np.argmin(C[i, disp_j])]
            val = min(s_copy[i], d_copy[mejor_j])
            x[i, mejor_j] = val
            s_copy[i] -= val
            d_copy[mejor_j] -= val
    return asegurar_m_n_1(x, S, D)

def aproximacion_vogel(S, D, C):
    x = np.zeros((len(S), len(D)))
    s_copy, d_copy = S.copy(), D.copy()
    filas_activas, cols_activas = list(range(len(S))), list(range(len(D)))
    
    while filas_activas and cols_activas:
        p_f, p_c = [], []
        for i in filas_activas:
            costos = sorted([C[i, j] for j in cols_activas])
            p_f.append((costos[1]-costos[0] if len(costos)>1 else costos[0], 'f', i))
        for j in cols_activas:
            costos = sorted([C[i, j] for i in filas_activas])
            p_c.append((costos[1]-costos[0] if len(costos)>1 else costos[0], 'c', j))
            
        max_p = max(p_f + p_c, key=lambda x: x[0])
        if max_p[1] == 'f':
            i = max_p[2]
            j = cols_activas[np.argmin([C[i, c] for c in cols_activas])]
        else:
            j = max_p[2]
            i = filas_activas[np.argmin([C[r, j] for r in filas_activas])]
            
        val = min(s_copy[i], d_copy[j])
        x[i, j] = val
        s_copy[i] -= val
        d_copy[j] -= val
        if s_copy[i] < 1e-9: filas_activas.remove(i)
        elif d_copy[j] < 1e-9: cols_activas.remove(j)
            
    return asegurar_m_n_1(x, S, D)

def obtener_base(x):
    # Consideramos b?sica cualquier celda con valor > 1e-11 (incluye ceros t?cnicos)
    return [(i, j) for i in range(x.shape[0]) for j in range(x.shape[1]) if x[i, j] > 1e-11]

def calcular_duales(base, C):
    m, n = C.shape
    u, v = {0: 0.0}, {}
    # Resolvemos el sistema u_i + v_j = C_ij
    for _ in range(m + n):
        for r, c in base:
            if r in u and c not in v:
                v[c] = float(C[r, c] - u[r])
            elif c in v and r not in u:
                u[r] = float(C[r, c] - v[c])
    return (np.array([u.get(i, 0.0) for i in range(m)]), 
            np.array([v.get(j, 0.0) for j in range(n)]))

def obtener_ciclo(base, inicio):
    # Algoritmo de b?squeda de camino cerrado (Backtracking)
    def buscar(actual, camino, fila_sig):
        if len(camino) > 3:
            if fila_sig and actual[0] == inicio[0]: return camino
            if not fila_sig and actual[1] == inicio[1]: return camino
        
        for sig in base:
            if sig not in camino:
                if fila_sig and sig[0] == actual[0]:
                    res = buscar(sig, camino + [sig], False)
                    if res: return res
                if not fila_sig and sig[1] == actual[1]:
                    res = buscar(sig, camino + [sig], True)
                    if res: return res
        return None
    return buscar(inicio, [inicio], True)

# --- 2. INTERFAZ STREAMLIT ---

st.set_page_config(page_title="Optimizador de Transporte", layout="wide")
st.title("Optimizador de Transporte (A prueba de Degeneración)")

with st.sidebar:
    st.header("1. Ajustes")
    metodo = st.selectbox("Método de Solución Inicial", ["Esquina Noroeste", "Costo Mínimo por Fila", "Aproximación de Vogel"])
    st.divider()
    st.header("2. Dimensiones")
    n_or = st.number_input("Orígenes (Filas)", min_value=2, max_value=10, value=3)
    n_de = st.number_input("Destinos (Columnas)", min_value=2, max_value=10, value=4)

if n_or == 3 and n_de == 4:
    oferta_def, demanda_def = [20, 30, 25], [10, 25, 20, 20]
    costos_def = [[8, 6, 10, 9], [9, 7, 4, 2], [3, 4, 2, 5]]
else:
    oferta_def, demanda_def = [10] * n_or, [10] * n_de
    costos_def = np.zeros((n_or, n_de))

c1, c2, c3 = st.columns([1, 1, 3])
with c1: of_df = st.data_editor(pd.DataFrame(oferta_def, columns=["Oferta"], index=[f"O{i+1}" for i in range(n_or)]))
with c2: de_df = st.data_editor(pd.DataFrame(demanda_def, columns=["Demanda"], index=[f"D{j+1}" for j in range(n_de)]))
with c3: c_df = st.data_editor(pd.DataFrame(costos_def, columns=[f"D{j+1}" for j in range(n_de)], index=[f"O{i+1}" for i in range(n_or)]))

if st.button("Resolver Problema"):
    S, D, C = of_df["Oferta"].values, de_df["Demanda"].values, c_df.values
    
    suma_s, suma_d = sum(S), sum(D)
    
    # Alerta visual si el problema no est? balanceado
    if suma_s != suma_d:
        st.warning(f" El problema NO está balanceado (Oferta: {suma_s} | Demanda: {suma_d}). Se ha añadido un nodo ficticio con costo 0 autom?ticamente.")
    else:
        st.success(" El problema está perfectamente balanceado.")
        
    S_b, D_b, C_b = balancear_problema(S, D, C)
    
    lbl_r = [f"O{i+1}" for i in range(len(S))] + (["O_Ficticio"] if len(S_b) > len(S) else [])
    lbl_c = [f"D{j+1}" for j in range(len(D))] + (["D_Ficticio"] if len(D_b) > len(D) else [])

    if metodo == "Esquina Noroeste":
        x = esquina_noroeste(S_b, D_b)
    elif metodo == "Costo Mínimo por Fila":
        x = minimo_costo_fila(S_b, D_b, C_b)
    else:
        x = aproximacion_vogel(S_b, D_b, C_b)
        
    st.subheader(f" Solución Inicial ({metodo})")
    st.dataframe(pd.DataFrame(np.round(x), index=lbl_r, columns=lbl_c).astype(int))
    st.metric("Costo Inicial", f"{int(np.sum(np.round(x) * C_b))} €")
    
    for it in range(1, 21):
        base = obtener_base(x)
        u, v = calcular_duales(base, C_b)
        
        cr = np.zeros(C_b.shape)
        min_cr, entrante = 0, None
        
        for r in range(C_b.shape[0]):
            for c in range(C_b.shape[1]):
                if (r, c) not in base:
                    cr[r, c] = C_b[r, c] - u[r] - v[c]
                    if cr[r, c] < min_cr - 1e-6:
                        min_cr, entrante = cr[r, c], (r, c)
        
        # UI: Mostrar Variables Duales y Costos Relativos
        with st.expander(f" Iteración {it} | Costo Actual: {int(np.sum(np.round(x)*C_b))} €"):
            
            # ---> L?NEAS A?ADIDAS: Mostrar la matriz de asignaci?n actual <---
            st.write("**Volúmenes Asignados:**")
            st.dataframe(pd.DataFrame(np.round(x), index=lbl_r, columns=lbl_c).astype(int))
            # ----------------------------------------------------------------
            
            col_u, col_v = st.columns(2)
            
            with col_u:
                st.write("**Duales de Fila ($u_i$):**")
                st.dataframe(pd.DataFrame(np.round(u, 2), index=lbl_r, columns=["u"]))
                
            with col_v:
                st.write("**Duales de Columna ($v_j$):**")
                st.dataframe(pd.DataFrame(np.round(v, 2), index=lbl_c, columns=["v"]).T)
                
            st.write("**Costos Relativos ($C_{ij} - u_i - v_j$):**")
            st.dataframe(pd.DataFrame(np.round(cr, 2), index=lbl_r, columns=lbl_c).style.highlight_min(axis=None, color='#ff9999'))
        
        if entrante is None:
            st.success(f" Solución óptima Alcanzada! Costo Total Mínimo: **{int(np.sum(np.round(x)*C_b))} €**")
            st.table(pd.DataFrame(np.round(x), index=lbl_r, columns=lbl_c).astype(int))
            break
            
        ciclo = obtener_ciclo(base + [entrante], entrante)
        nodos_neg = [ciclo[k] for k in range(1, len(ciclo), 2)]
        theta = min([x[r, c] for r, c in nodos_neg])
        
        for k, (r, c) in enumerate(ciclo):
            if k % 2 == 0: x[r, c] += theta
            else: x[r, c] -= theta