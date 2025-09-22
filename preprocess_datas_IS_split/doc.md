# Pre-processing, Importance Sampling, and CSV File Management

This repository focuses on handling observations and preparing datasets for training models, particularly those involving precipitation data. One crucial step in this process is **Importance Sampling (IS)**.

---

## Importance Sampling (IS)

Inspired by  Ravuri et al. (2021) dans [*Skilful precipitation nowcasting using deep generative models of radar*](https://doi.org/10.1038/s41586-021-03854-z), Importance Sampling helps avoid over-representing samples with little or no precipitation in the dataset. By assigning an "importance" score to each sample based on its contribution to the training process, we can prioritize the most significant data points. 

This method ensures that the network learns to better reproduce precipitation patterns by focusing on samples with meaningful precipitation data.

---

## Overview of the Workflow

This folder contains several Python scripts for dataset preparation, each performing specific operations. Below is a step-by-step explanation of the process:

### 1. Merge Samples into a Giga File

Using the `method_type=merge_into_giga_file` parameter in `main.py`, all samples are merged into a single large file (to simplify the processing). This process outputs:
- A folder containing a CSV file.
- A corresponding giga file.

### 2. Calculate Importance Sampling

With the `method_type=importance_sampling` parameter in `main.py`, we calculate the importance of each sample. The process:
- Adds a new column, `importance`, to the CSV file.
- Filters out less significant samples, retaining only those seemed important for training.

The resulting CSV file is ready for further processing.

### 3. Apply Bootstrap Sampling (Optional)

By toggling the bootstrap parameter in `main.py`, you can enable bootstrap sampling:
- This method performs **n_bootstrap Importance samplings** on the entire dataset.
- It generates **n_bootstrap CSV files**, each containing different subsets of the original dataset.
- Finally it merges theses **n_bootstrap** csv files in one final without duplicates. 

### 4. Select Ensemble Members

After importance sampling and optional bootstrap, we ensure that for a given lead time and date, all ensemble members are included in the dataset.
- For exemples, if **n_ensemble**  members of an ensemble are found in the current CSV file, the method selects all 16 members from the original dataset.

### 5. Split the Dataset

Finally, the processed dataset is split into three separate CSV files for:
- **Training**
- **Validation**
- **Testing**

---

### 6. Compute stat

If you want to compare the number of pixels with precipitation greater than or equal to a list of thresholds for each CSV file obtained, use the **stat** parameter.

## Summary

This repository provides a structured approach to preparing datasets for StyleGAN. By using importance sampling, optional bootstrap sampling, and member selection, we ensure a well-balanced and meaningful dataset. The final splits enable effective model training, validation, and testing.
