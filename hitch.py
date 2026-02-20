import streamlit as st
import numpy as np
import pandas as pd

# --- MÉTODOS DE ASIGNACIÓN INICIAL ---

def inicial_esquina_noroeste(s, d):
    s_tmp, d_tmp = s.copy(), d.copy()
    x = np.zeros((len(s), len(d)))
    i, j = 0, 0
    while i < len(s) and j < len(d):
        cant = min(s_tmp[i], d_tmp[j])
        x[i, j] = cant
        s_tmp[i] -= cant
        d_tmp[j] -= cant
        if s_tmp[i] == 0: i += 1
        elif d_tmp[j] == 0: j += 1
    return x

def inicial_minimo_filas(s, d, costos):
    s_tmp, d_tmp = s.copy(), d.copy()
    x = np.zeros(costos.shape)
    for i in range(len(s)):
        while s_tmp[i] > 0:
            # Buscar destinos con demanda pendiente
            disp = [j for j in range(len(d)) if d_tmp[j] > 0]
            if not disp: break
            # Encontrar el de menor costo en la fila actual i
            j = disp[np.argmin(costos[i, disp])]
            cant = min(s_tmp[i], d_tmp[j])
            x[i, j] = cant
            s_tmp[i] -= cant
            d_tmp[j] -= cant
    return x

def inicial_vogel(s, d, costos):
    s_tmp, d_tmp = s.copy(), d.copy()
    x = np.zeros(costos.shape)
    f_vivas, c_vivas = list(range(len(s))), list(range(len(d)))
    while f_vivas and c_vivas:
        penal = []
        for i in f_vivas:
            c_f = sorted([costos[i, j] for j in c_vivas])
            penal.append((c_f[1]-c_f[0] if len(c_f)>1 else c_f[0], 'F', i))
        for j in c_vivas:
            c_c = sorted([costos[i, j] for i in f_vivas])
            penal.append((c_c[1]-c_c[0] if len(c_c)>1 else c_c[0], 'C', j))
        p_max, tipo, idx = max(penal, key=lambda x: x[0])
        if tipo == 'F':
            i = idx
            j = c_vivas[np.argmin([costos[i, c] for c in c_vivas])]
        else:
            j = idx
            i = f_vivas[np.argmin([costos[r, j] for r in f_vivas])]
        cant = min(s_tmp[i], d_tmp[j])
        x[i, j] = cant
        s_tmp[i], d_tmp[j] = s_tmp[i]-cant, d_tmp[j]-cant
        if s_tmp[i] == 0: f_vivas.remove(i)
        elif d_tmp[j] == 0: c_vivas.remove(j)
    return x

# --- LÓGICA MODI Y BALANCEO ---

def balancear_problema(S, D, C):
    S, D = S.astype(float), D.astype(float)
    t_s, t_d = sum(S), sum(D)
    new_S, new_D, new_C = S.copy(), D.copy(), C.copy()
    msg = ""
    if t_s > t_d:
        diff = t_s - t_d
        new_D = np.append(D, diff)
        new_C = np.hstack((C, np.zeros((len(S), 1))))
        msg = f"⚖️ **Balanceo:** Oferta > Demanda. Se añadió Destino Ficticio (+{diff})."
    elif t_d > t_s:
        diff = t_d - t_s
        new_S = np.append(S, diff)
        new_C = np.vstack((C, np.zeros((1, len(D)))))
        msg = f"⚖️ **Balanceo:** Demanda > Oferta. Se añadió Origen Ficticio (+{diff})."
    else: msg = "✅ El problema ya está balanceado."
    return new_S, new_D, new_C, msg

def calcular_modi(x, costos):
    nf, nc = costos.shape
    u, v = [None]*nf, [None]*nc
    u[0] = 0
    for _ in range(nf + nc):
        for r in range(nf):
            for c in range(nc):
                if x[r, c] > 0:
                    if u[r] is not None and v[c] is None: v[c] = costos[r, c] - u[r]
                    elif v[c] is not None and u[r] is None: u[r] = costos[r, c] - v[c]
    u = np.array([val if val is not None else 0 for val in u])
    v = np.array([val if val is not None else 0 for val in v])
    cr = np.zeros((nf, nc))
    for r in range(nf):
        for c in range(nc):
            if x[r, c] == 0: cr[r, c] = costos[r, c] - u[r] - v[c]
    return u, v, cr

def encontrar_ciclo(x, inicio):
    nf, nc = x.shape
    def buscar(camino, fila_act):
        if len(camino) > 3:
            if fila_act and camino[-1][0] == camino[0][0]: return camino
            if not fila_act and camino[-1][1] == camino[0][1]: return camino
        r, c = camino[-1]
        if fila_act:
            for j in range(nc):
                if j != c and x[r, j] > 0:
                    res = buscar(camino + [(r, j)], False)
                    if res: return res
        else:
            for i in range(nf):
                if i != r and x[i, c] > 0:
                    res = buscar(camino + [(i, c)], True)
                    if res: return res
        return None
    return buscar([inicio], False)

# --- INTERFAZ STREAMLIT ---

st.set_page_config(page_title="Hitchcock Solver Full", layout="wide")
st.title("🚛 Solucionador de Transporte Hitchcock")

with st.sidebar:
    st.header("Ajustes")
    metodo_sel = st.selectbox("Método Inicial", ["Esquina Noroeste", "Mínimo por Fila", "Vogel"])
    n_or = st.number_input("Orígenes", 2, 6, 3)
    n_de = st.number_input("Destinos", 2, 6, 4)

st.subheader("1. Entrada de Datos")
col1, col2, col3 = st.columns([1, 1, 3])
with col1: of_df = st.data_editor(pd.DataFrame({"Oferta": [20, 30, 25, 10, 10, 10][:n_or]}))
with col2: de_df = st.data_editor(pd.DataFrame({"Demanda": [10, 25, 20, 20, 10, 10][:n_de]}))
with col3: 
    c_base = np.array([[8,6,10,9,5,5],[9,7,4,2,5,5],[3,4,2,5,5,5]])[:n_or, :n_de]
    c_df = st.data_editor(pd.DataFrame(c_base, index=[f"O{i+1}" for i in range(n_or)], columns=[f"D{j+1}" for j in range(n_de)]))

if st.button("🚀 Resolver Paso a Paso"):
    S, D, C = of_df["Oferta"].values, de_df["Demanda"].values, c_df.values
    S_b, D_b, C_b, msg = balancear_problema(S, D, C)
    st.info(msg)
    
    lbl_r = [f"O{i+1}" for i in range(len(S))] + (["O_fict"] if len(S_b) > len(S) else [])
    lbl_c = [f"D{j+1}" for j in range(len(D))] + (["D_fict"] if len(D_b) > len(D) else [])

    # Solución Inicial
    if metodo_sel == "Esquina Noroeste": x = inicial_esquina_noroeste(S_b, D_b)
    elif metodo_sel == "Mínimo por Fila": x = inicial_minimo_filas(S_b, D_b, C_b)
    else: x = inicial_vogel(S_b, D_b, C_b)
    
    st.write(f"### 🏁 Fase 1: Solución por {metodo_sel}")
    st.dataframe(pd.DataFrame(x, index=lbl_r, columns=lbl_c))
    st.metric("Costo Inicial", f"{np.sum(x*C_b)} €")

    # Iteraciones MODI
    it = 1
    while it <= 12:
        u, v, cr = calcular_modi(x, C_b)
        min_v = np.min(cr)
        
        if min_v >= -1e-9: # Tolerancia para flotantes
            st.success(f"🏆 ¡Solución Óptima alcanzada en la iteración {it-1}!")
            st.write("### Asignación Final Óptima")
            st.table(pd.DataFrame(x, index=lbl_r, columns=lbl_c))
            st.metric("Costo Mínimo Total", f"{np.sum(x*C_b)} €")
            break
            
        entrada = np.unravel_index(np.argmin(cr), cr.shape)
        
        with st.expander(f"🔍 Paso {it}: Análisis de Duales y Costes Relativos"):
            ca, cb = st.columns(2)
            with ca:
                st.write("**Multiplicadores ($u_i, v_j$):**")
                st.write(pd.DataFrame({"u": u}, index=lbl_r).T)
                st.write(pd.DataFrame({"v": v}, index=lbl_c).T)
            with cb:
                st.write(f"**Costo actual:** {np.sum(x*C_b)} €")
                st.write(f"**Entra celda:** {lbl_r[entrada[0]]}-{lbl_c[entrada[1]]}")

            st.write("**Matriz de Costes Relativos ($\bar{C}_{ij}$):**")
            st.dataframe(pd.DataFrame(cr, index=lbl_r, columns=lbl_c).style.highlight_min(axis=None, color='lightcoral'))

        ciclo = encontrar_ciclo(x, entrada)
        n_neg = [ciclo[k] for k in range(1, len(ciclo), 2)]
        theta = min(x[r, c] for r, c in n_neg)
        for k, (r, c) in enumerate(ciclo):
            x[r, c] += theta if k % 2 == 0 else -theta
        it += 1