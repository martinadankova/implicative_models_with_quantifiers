*Implicative Models with Quantifiers: A Fuzzy Set Approach to Interpretable Modeling.*

### 🔍 Overview

This repository demonstrates the construction of fuzzy implicative models based on functional data, using **fuzzy sets**, **fuzzy quantifiers**, and **linguistic implications**. The goal is to model relationships between input-output pairs using an interpretable fuzzy rule base enhanced by quantifier-driven confidence.

### 📂 Structure

* `definitions/` – Core implementation modules:

  * `FuzzyRelations.py`: fuzzy sets, fuzzy implications, interval functions
  * `quantifiers.py`: linguistic quantifiers and truth evaluation
  * `models.py`: visualization, rule generation, Mamdani and implicative simulation
* `data/funcnihodnoty.xlsx` – Input data (two columns: `x` and `fx`)
* `Quantifier_Based_Implicative_Models.ipynb` the main script – Construction and visualization of the implicative fuzzy model
* `Quantifiers_Examples.ipynb` - shows examples of implicative quantifier values 
---

### 📈 What the Main Script `Quantifier_Based_Implicative_Models` Does

1. **Loads functional data** (`x`, `fx`) from Excel.
2. **Generates fuzzy partitions** on `X` and `Y` domains based on quantifiers.
3. **Builds fuzzy rules** using:

   * Fuzzy intervals (`Aᵢ`, `Bᵢ`) for inputs/outputs
   * Confidence measures via 2×2 contingency tables
   * Quantified implications (e.g., $Q(A \Rightarrow B)$)
4. **Visualizes**:

   * Fuzzy sets on `X`, `Y`
   * Rule surfaces
   * Resulting quantified implicative model as fuzzy relation over (X, Y)

---

### ⚙️ Requirements

* Python 3.8+
* Libraries:

  ```bash
  pip install numpy pandas matplotlib openpyxl plotly
  ```

---

### 🚀 Running the Model

```bash
python main.py  # or run in a Jupyter notebook
```

Make sure the Excel file `funcnihodnoty.xlsx` is in `data/`.

---

### 🧠 Example Use Case

This model can be used to:

* Analyze and simulate fuzzy control systems based on human-like rules.
* Construct interpretable regression surfaces from empirical data.
* Compare Mamdani vs implicative rule inference.

---

### 📊 Output

* Fuzzy membership functions (Aᵢ and Bᵢ)
* 2D surface of implication model
* Quantifier confidence values per rule
* Visual comparison of implicative vs Mamdani models

---

### 🧩 References

This approach is based on fuzzy quantifier-based implication modeling and rule construction. See topics on:

* Fuzzy quantification
* GUHA method extensions
* Interpretable fuzzy modeling

📚 Citation
If you use this code for academic purposes, please cite it as:

Martina Daňková. Implicative Models with Quantifiers: A Fuzzy Set Approach to Interpretable Modeling. GitHub, 2025. https://github.com/martinadankova/implicative_models_with_quantifiers

If part of a paper, you may cite the underlying methods from:

* Martina Daňková, Weighted Fuzzy Rules Based on Implicational Quantifiers.
Integrated Uncertainty in Knowledge Modelling and Decision Making: 10th International Symposium, IUKM 2023, Kanazawa, Japan, November 2–4, 2023, Proceedings, Part I, Pages 27 - 36.
https://doi.org/10.1007/978-3-031-46775-2_3

bibitem: 

@inproceedings{10.1007/978-3-031-46775-2_3,
author = {Da\v{n}kov\'{a}, Martina},
title = {Weighted Fuzzy Rules Based on Implicational Quantifiers},
year = {2023},
isbn = {978-3-031-46774-5},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-46775-2_3},
doi = {10.1007/978-3-031-46775-2_3},
abstract = {In this paper, we explore the use of General Unary Hypotheses Automaton (GUHA) quantifiers, explicitly implicational quantifiers, for analyzing specific relational dependencies. We discuss their suitability in fuzzy modeling and demonstrate their integration with appropriate fuzzy rules to create a new class of weighted fuzzy rules. This study contributes to the advancement of fuzzy modeling and offers a framework for further research and practical applications.},
booktitle = {Integrated Uncertainty in Knowledge Modelling and Decision Making: 10th International Symposium, IUKM 2023, Kanazawa, Japan, November 2–4, 2023, Proceedings, Part I},
pages = {27–36},
numpages = {10},
keywords = {Implicational Quantifiers, IF–THEN Rules, Fuzzy Logic, Weighted Fuzzy Rules},
location = {Kanazawa, Japan}
}

* A preprint of the full paper containing the whole material implemented here is by
Daňková, Martina and Hliněná, Dana, Fuzzy Rules with Quantifiers as Weights. Available at SSRN: http://dx.doi.org/10.2139/ssrn.5169367
