import streamlit as st
import pandas as pd
import subprocess
import os
from datetime import datetime
import numpy as np

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Quantum Random Number Generator — Single Qubit",
    layout="wide",
    page_icon="⚛️"
)

# ---------------------- HEADER ----------------------
st.markdown("<h1 style='text-align:center;'>⚛️ Quantum Random Number Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Generate <b>true quantum random bits</b> using single-qubit measurement simulation or real hardware.</p>", unsafe_allow_html=True)

with st.expander("ℹ️ About QRNG"):
    st.markdown("""
    The **Quantum Random Number Generator (QRNG)** uses the laws of quantum mechanics  
    to produce truly unpredictable random bits.  
                
    You can select from three backends:

    - 🧮 **Simulator** → Fast local quantum circuit simulation  
    - 🧊 **Pseudo** → Classical fake randomness for quick testing  
    - 🧠 **IBMQ** → Real quantum hardware from IBM Quantum Experience  
    """)

# ---------------------- SIDEBAR CONTROLS ----------------------
st.sidebar.header("⚙️ Generator Settings")

n_bits = st.sidebar.number_input("Number of Bits", min_value=8, max_value=65536, value=1024, step=8)
backend = st.sidebar.selectbox("Backend", ("simulator", "pseudo", "ibmq"))

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = os.path.join(output_dir, f"qrng_bits_{timestamp}.csv")

# ---------------------- GENERATE BUTTON ----------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_btn = st.button("🎲 Generate Quantum Random Bits", use_container_width=True)

if generate_btn:
    st.info(f"Running QRNG for **{n_bits} bits** using `{backend}` backend...")

    cmd = f"python scripts/generate_qrnbits.py --n_bits {n_bits} --backend {backend} --out {out_file}"

    with st.spinner("⏳ Quantum computation in progress..."):
        try:
            subprocess.check_call(cmd, shell=True)
            df = pd.read_csv(out_file)

            st.success(f"✅ Successfully generated {len(df)} quantum random bits!")

            # ---------------------- VISUALIZATION ----------------------
            st.markdown("### 📊 Bit Distribution Overview")
            bit_counts = df['bit'].value_counts().sort_index()
            st.bar_chart(bit_counts)

            # ---------------------- RANDOMNESS METRICS ----------------------
            colA, colB, colC = st.columns(3)
            prob_one = df['bit'].mean()
            entropy = -(
                (prob_one * np.log2(prob_one + 1e-9))
                + ((1 - prob_one) * np.log2(1 - prob_one + 1e-9))
            )

            colA.metric("Estimated P(1)", f"{prob_one:.4f}")
            colB.metric("Entropy (bits)", f"{entropy:.4f}")
            colC.metric("Total Bits", f"{len(df)}")

            # ---------------------- DOWNLOAD ----------------------
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Generated Bits (CSV)",
                data=csv_data,
                file_name=f"qrng_bits_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # ---------------------- RAW PREVIEW ----------------------
            with st.expander("🧾 View Raw Bit Data"):
                st.dataframe(df.head(20))

        except subprocess.CalledProcessError as e:
            st.error(f"⚠️ Generation failed:\n{e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ---------------------- FOOTER ----------------------
st.markdown("""
---
<p style='text-align:center;'>
<b>Developed by:</b> SUJAL 🚀  
<b>Tech Stack:</b> Python · Qiskit · Streamlit  
<br>
<i>Quantum randomness through single-qubit measurements in |+⟩ basis.</i>
</p>
""", unsafe_allow_html=True)
