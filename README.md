# Modeling Influenza and RSV-attributable ILI Repository

This repository contains the dataset and Python programs used in the analysis of influenza-like illness (ILI) attributable to influenza and respiratory syncytial virus (RSV).

## Contents

### Data
- flu_rsv.xlsx  
  This file contains:
  - Virus positivity rates  
  - ILI case counts  
  - Population data  

### Code
The following Python scripts were used to run the three models considered in the study:
- modeling_ILI_M1.py – Model 1  
- modeling_ILI_M2.py – Model 2  
- modeling_ILI_M3.py – Model 3  

## How to Run the Models

1. Download or clone this repository.
2. Open any of the Python scripts:
   - modeling_ILI_M1.py  
   - modeling_ILI_M2.py  
   - modeling_ILI_M3.py  
3. Update the file path to the dataset in the script before running the model. Example:
   data = pd.read_excel("your/local/path/flu_rsv.xlsx")
4. Run the script in Python.

## Important Note
Before running the models, make sure to change the filename path in the code so that it correctly points to the location of the data file on your computer.
