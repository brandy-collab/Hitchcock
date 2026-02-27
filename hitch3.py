import streamlit as st
import numpy as np
import pandas as pd

# --- 1. ALGORITMOS PRINCIPALES (Sin cambios en la lógica base) ---

def balancear_problema(S, D, C):
    S, D = S.astype(float), D.astype(float)
    if sum(S) > sum(D):
        D = np.append(D, sum(S) - sum(D))
        C = np.hstack((C, np.zeros((len(S), 1))))
    elif sum(D) > sum(S):
        S = np.append(S, sum(D) - sum(S))
        C = np.vstack((C, np.zeros((1, len(D)))))
        
    eps = 1e-5
    for i in range(len(S)):
        S[i] += eps
        D[-1] += eps
        
    return S, D, C

def esquina_noroeste(S, D):
    x = np.zeros((len(S), len(D)))
    s_copy, d_copy = S.copy(), D.copy()
    i, j = 0, 0
    while i < len(S) and j < len(D):
        val = min(s_copy[i], d_copy[j])
        x[i, j] = val
        s_copy[i] -= val
        d_copy[j] -= val
        if s_copy[i] < 1e-9 and d_copy[j] < 1e-9:
            i += 1
            j += 1
        elif s_copy[i] < 1e-9: 
            i += 1
        else: 
            j += 1
    return x

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
    return x

def aproximacion_vogel(S, D, C):
    x = np.zeros((len(S), len(D)))
    s_copy, d_copy = S.copy(), D.copy()
    filas_activas, cols_activas = list(range(len(S))), list(range(len(D)))
    while filas_activas and cols_activas:
        penal_filas, penal_cols = [], []
        for i in filas_activas:
            costos = sorted([C[i, j] for j in cols_activas])
            pen = costos[1] - costos[0] if len(costos) > 1 else costos[0]
            penal_filas.append((pen, 'f', i))
        for j in cols_activas:
            costos = sorted([C[i, j] for i in filas_activas])
            pen = costos[1] - costos[0] if len(costos) > 1 else costos[0]
            penal_cols.append((pen, 'c', j))
        max_pen = max(penal_filas + penal_cols, key=lambda item: item[0])
        if max_pen[1] == 'f':
            i = max_pen[2]
            j = cols_activas[np.argmin([C[i, c] for c in cols_activas])]
        else:
            j = max_pen[2]
            i = filas_activas[np.argmin([C[r, j] for r in filas_activas])]
        val = min(s_copy[i], d_copy[j])
        x[i, j] = val
        s_copy[i] -= val
        d_copy[j] -= val
        if s_copy[i] < 1e-9 and d_copy[j] < 1e-9:
            if len(filas_activas) > 1: filas_activas.remove(i)
            else: cols_activas.remove(j)
        elif s_copy[i] < 1e-9: filas_activas.remove(i)
        else: cols_activas.remove(j)
    return x

def obtener_base(x):
    return [(i, j) for i in range(x.shape[0]) for j in range(x.shape[1]) if x[i, j] > 1e-8]

def calcular_duales(base, C):
    m, n = C.shape
    u, v = {0: 0.0}, {}
    progreso = True
    while progreso and (len(u) < m or len(v) < n):
        progreso = False
        for r, c in base:
            if r in u and c not in v:
                v[c] = C[r, c] - u[r]
                progreso = True
            elif c in v and r not in u:
                u[r] = C[r, c] - v[c]
                progreso = True
    return (np.array([u.get(i, 0.0) for i in range(m)]), np.array([v.get(j, 0.0) for j in range(n)]))

def obtener_ciclo(celdas_base, celda_inicio):
    pila = [(celda_inicio, True, [celda_inicio]), (celda_inicio, False, [celda_inicio])]
    while pila:
        actual, mover_fila, camino = pila.pop()
        if len(camino) >= 4:
            if mover_fila and actual[0] == celda_inicio[0]: return camino
            if not mover_fila and actual[1] == celda_inicio[1]: return camino
        r, c = actual
        for br, bc in celdas_base:
            if (br, bc) not in camino:
                if mover_fila and br == r and bc != c:
                    pila.append(((br, bc), False, camino + [(br, bc)]))
                elif not mover_fila and bc == c and br != r:
                    pila.append(((br, bc), True, camino + [(br, bc)]))
    return None

# --- 3. FUNCIÓN DE ESTILO PARA RESALTAR LA BASE ---

def resaltar_base(df, celdas_base):
    """
    Pinta de amarillo las celdas que pertenecen a la base.
    """
    estilo = pd.DataFrame('', index=df.index, columns=df.columns)
    for r, c in celdas_base:
        estilo.iloc[r, c] = 'background-color: yellow; color: black; font-weight: bold'
    return estilo

# --- 4. INTERFAZ STREAMLIT ---

st.set_page_config(page_title="Optimizador de Transporte", layout="wide")
st.title("🚛 Optimizador de Transporte")
st.info("Nota: Las celdas resaltadas en **amarillo** representan las variables básicas de la solución (la base).")

with st.sidebar:
    st.header("1. Ajustes")
    metodo = st.selectbox("Método de Solución Inicial", ["Esquina Noroeste", "Costo Mínimo por Fila", "Aproximación de Vogel"])
    st.divider()
    st.header("2. Dimensiones")
    n_or = st.number_input("Orígenes", min_value=2, max_value=10, value=3)
    n_de = st.number_input("Destinos", min_value=2, max_value=10, value=4)

# Valores por defecto
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

if st.button("🚀 Resolver Problema"):
    S, D, C = of_df["Oferta"].values, de_df["Demanda"].values, c_df.values
    S_b, D_b, C_b = balancear_problema(S, D, C)
    
    lbl_r = [f"O{i+1}" for i in range(len(S))] + (["O_Ficticio"] if len(S_b) > len(S) else [])
    lbl_c = [f"D{j+1}" for j in range(len(D))] + (["D_Ficticio"] if len(D_b) > len(D) else [])

    # Solución Inicial
    if metodo == "Esquina Noroeste": x = esquina_noroeste(S_b, D_b)
    elif metodo == "Costo Mínimo por Fila": x = minimo_costo_fila(S_b, D_b, C_b)
    else: x = aproximacion_vogel(S_b, D_b, C_b)
        
    st.subheader(f"🏁 Solución Inicial ({metodo})")
    df_init = pd.DataFrame(np.round(x), index=lbl_r, columns=lbl_c).astype(int)
    base_init = obtener_base(x)
    st.dataframe(df_init.style.apply(resaltar_base, celdas_base=base_init, axis=None))
    
    # Iteraciones de Optimización
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
        
        with st.expander(f"🔄 Iteración {it} | Costo: {int(np.sum(np.round(x)*C_b))} €"):
            st.write("**Asignación Actual (Amarillo = Base):**")
            df_iter = pd.DataFrame(np.round(x), index=lbl_r, columns=lbl_c).astype(int)
            st.dataframe(df_iter.style.apply(resaltar_base, celdas_base=base, axis=None))
            
            col_u, col_v = st.columns(2)
            with col_u: st.dataframe(pd.DataFrame(np.round(u, 2), index=lbl_r, columns=["u"]))
            with col_v: st.dataframe(pd.DataFrame(np.round(v, 2), index=lbl_c, columns=["v"]).T)
            
            st.write("**Costos Relativos:**")
            st.dataframe(pd.DataFrame(np.round(cr, 2), index=lbl_r, columns=lbl_c).style.highlight_min(axis=None, color='#ff9999'))
        
        if entrante is None:
            st.success(f"🏆 ¡Óptimo alcanzado! Costo: {int(np.sum(np.round(x)*C_b))} €")
            df_final = pd.DataFrame(np.round(x), index=lbl_r, columns=lbl_c).astype(int)
            st.dataframe(df_final.style.apply(resaltar_base, celdas_base=base, axis=None))
            break
            
        ciclo = obtener_ciclo(base + [entrante], entrante)
        nodos_neg = [ciclo[k] for k in range(1, len(ciclo), 2)]
        theta = min([x[r, c] for r, c in nodos_neg])
        for k, (r, c) in enumerate(ciclo):
            if k % 2 == 0: x[r, c] += theta
            else: x[r, c] -= theta
