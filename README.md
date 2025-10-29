# ⚛️ Quantum Random Number Generator — Single Qubit

---

## 🎯 Overview
The **Quantum Random Number Generator (QRNG)** uses the principles of **quantum mechanics** to produce *truly random bits*.  
It prepares a **single qubit in superposition** and measures it, producing inherently unpredictable results — ideal for **cryptography, simulations, and secure computations**.

This project includes:
- A **Streamlit web dashboard** for bit generation and visualization  
- Python scripts for **entropy analysis**, **randomness extraction**, and **validation tests**
---

## 🧩 Project Structure
<pre>

random_number_generator/
│
├── app/
│ └── streamlit_app.py # Streamlit web dashboard for QRNG
│
├── data/
│ ├── qrng_bits_*.csv # Generated random bits data files
│ └── qrng_bits.metajson # Metadata of generation sessions
│
├── scripts/
│ ├── generate_qrnbits.py # Main script for quantum random bit generation
│ ├── entropy.py # Calculates entropy of generated bits
│ ├── extractor.py # Performs randomness extraction
│ └── randomness_test.py # Statistical randomness verification tests
│
└── requirements.txt # Python dependencies
</pre>

---

## 🚀 Features
<pre>
✅ **Quantum Randomness** — Generated via single-qubit superposition and measurement  
✅ **Three Backends**
- 🧮 `simulator` → Fast local quantum simulation  
- 🧊 `pseudo` → Classical pseudo-random generation (for testing)  
- 🧠 `ibmq` → Real IBM Quantum hardware (requires IBMQ credentials)

✅ **Interactive Dashboard** — Built with Streamlit for real-time generation  
✅ **Entropy Calculation** — Evaluates randomness quality using Shannon entropy  
✅ **CSV Export** — Download and analyze generated bits  
✅ **Modern UI** — Clean, responsive interface with charts and metrics  
</pre>
---

## 🧠 How It Works
<pre>
1. **Initialize a single qubit** in state |0⟩  
2. Apply a **Hadamard gate** to create the superposition (|0⟩ + |1⟩) / √2  
3. **Measure** the qubit — collapses to either |0⟩ or |1⟩ randomly  
4. **Repeat** this process *n* times to form a random bit sequence  
5. Use optional **entropy** and **extractor** scripts for quality validation  
</pre>
---

## 🧰 Tech Stack
<pre>
- Python 
- Streamlit
- Qiskit
- Pandas
- NumPy
- Matplotlib 
</pre>
---

## ⚙️ Installation & Setup
<pre>
```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/random_number_generator.git
cd random_number_generator

# 2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate        # (Windows)
source venv/bin/activate     # (Linux/Mac)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit app
streamlit run app/streamlit_app.py
</pre>
---


## 🖥️ Usage
<pre>
Open the local URL (default: http://localhost:8501)

Choose the number of bits 

Select backend (simulator, pseudo, ibmq)

Click “Generate Quantum Random Bits”

View charts, bit distribution, and download CSV output
</pre>
---


## 🧪 Supporting Scripts
<pre>
Script	                         Description
generate_qrnbits.py:   |Core logic for quantum random number generation
entropy.py:	           |Calculates entropy and randomness score
extractor.py:	         |Cleans and extracts near-uniform random bits
randomness_test.py:	   |Performs NIST-like randomness tests
</pre>
---



## 🌟 Acknowledgements
<pre>
Qiskit — Quantum computing SDK by IBM

Streamlit — Python framework for interactive dashboards

Quantum mechanics fundamentals for true randomness generation
</pre>
---
