*Implicative Models with Quantifiers: A Fuzzy Set Approach to Interpretable Modeling.*

### 🔍 Overview

This repository demonstrates a construction of fuzzy implicative models based on functional data, using **fuzzy sets**, **fuzzy quantifiers**, and **linguistic implications**. The goal is to model relationships between input-output pairs using an interpretable fuzzy rule base enhanced by quantifier-driven confidence.

### 📂 Structure

* `definitions/` – Core implementation modules:

  * `FuzzyRelations.py`: fuzzy sets, fuzzy implications, interval functions
  * `quantifiers.py`: linguistic quantifiers and truth evaluation
  * `models.py`: visualization, rule generation, Mamdani and implicative simulation
* `data/
  * `funcnihodnoty.xlsx` – sythetic input data (two columns: `x` and `fx`)
  * `world_bank_indicators.csv` – real data gathered by World Bank 
* `Quantifier_Based_Implicative_Models.ipynb` the main script – Construction and visualization of the implicative fuzzy model
* `Quantifiers_Examples.ipynb` - shows examples of implicative quantifier values
* `WorldBankExample.ipynb` - provides example of building implicative model with quantifiers for world bank data set together with defuzzification methods and standard simple models
* `WorldBankExample10Rules.ipynb` - the same as above only the number of rules is set to 10
* `ParametersSavingUploading_QuantifiersBasedModels.ipynb` - provides a way how to store the model's parameters and load them for a future manipulations
  
---
### 🌍 Data Source
This project uses synthetic data and may be extended to real-world data such as the World Bank Development Indicators.

Example indicators used in the current code:

* GDP per capita (current US$) 

* Fertility rate, total (births per woman) 

Source: The World Bank – World Development Indicators
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

---
### 📈 What the Script `Quantifier_Based_Implicative_Models` Does

1. **Loads functional data** (`x`, `fx`) from Excel.
2. **Generates fuzzy partitions** on `X` and `Y` domains.
3. **Builds fuzzy rules** using:

   * Fuzzy intervals (`Aᵢ`, `Bᵢ`) for inputs/outputs
   * Confidence measures via 2×2 contingency tables
   * Quantified implications (e.g., $Q(A \Rightarrow B)$)
4. **Visualizes**:

   * Fuzzy sets on `X`, `Y`
   * Rule surfaces
   * Resulting quantified implicative model as fuzzy relation over (X, Y)

### 📈 What the Script `WorldBankExample` Does

1. **Loads the data** (`GDP per capita (current US$)`, `Fertility rate, total (births per woman)`) from csv.
2. **Generates fuzzy partitions** on `X` and `Y` domains.
3. **Builds fuzzy rules** using:

   * Fuzzy intervals (`Aᵢ`, `Bᵢ`) for inputs/outputs
   * Confidence measures via 2×2 contingency tables
   * Quantified implications (e.g., $Q(A \Rightarrow B)$)
4. **Visualizes**:

   * Fuzzy sets on `X`, `Y`
   * Rule surfaces
   * Resulting quantified implicative model as fuzzy relation over (X, Y)
   * Various defuzzification methods with visualisations together over the original data as well as the quantified implicative model

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

* A preprint of the full paper containing the whole material implemented here is by
Daňková, Martina and Hliněná, Dana, Fuzzy Rules with Quantifiers as Weights. Available at SSRN: http://dx.doi.org/10.2139/ssrn.5169367
