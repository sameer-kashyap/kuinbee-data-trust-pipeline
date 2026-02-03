# Kuinbee Data Trust Pipeline (KDTS – MVP)

This repository contains a **Python-based Data Trust Pipeline** built to evaluate the **credibility, safety, and usability of datasets** before they are listed on the Kuinbee marketplace.

The pipeline is inspired by Kuinbee’s **Dataset Credibility Checks (DCC)** framework and is designed as a **manual-to-automated MVP**, not a production system.

---

## 📌 Why this pipeline exists

Most data marketplaces focus on **distribution and access**.
Kuinbee focuses on **trust before access**.

Before selling a dataset, Kuinbee wants clear answers to questions like:
- Is this dataset technically sound?
- Does the data behave logically?
- Where did this data come from?
- Are we legally allowed to sell it?
- Will buyers actually be able to use it?

This pipeline exists to **systematically answer those questions** and convert them into a **single trust score: KDTS (Kuinbee Data Trust Score)**.

---

## 🧠 Core idea: Data Due Diligence (not just data cleaning)

This project treats dataset evaluation as **data due diligence**, similar to how financial or legal diligence is done — structured, explainable, and conservative.

Not all checks should be automated.
Some checks **must remain human-reviewed** by design.

---

## 🏗️ What this pipeline evaluates

The pipeline follows the same stages described in the **Dataset Credibility Checks (DCC)** document.

---

### 1️⃣ Parametric & Structural Checks (Machine-First)

**Question:** *Is this dataset technically sound?*

Automated checks include:
- Schema integrity (columns, data types, schema drift)
- Statistical sanity (ranges, impossible values, distributions)
- Completeness (missing values, critical fields)
- Uniqueness (duplicate rows, duplicate keys)

**Output:** Parametric Quality Score (0–100)

---

### 2️⃣ Consistency & Temporal Checks (Trust Layer)

**Question:** *Does this data behave logically over time and in context?*

Includes:
- Internal consistency (cross-column logic, units, referential integrity)
- Temporal validity (timestamps, gaps, surges — or explicit NA handling)
- Geographic consistency (only for geo datasets)

**Output:** Consistency risk flags and notes

---

### 3️⃣ Provenance & Methodology Checks (Human-Critical)

**Question:** *Where did this data actually come from?*

These checks are **documented, not blindly automated**:
- Source declaration (primary, licensed, scraped)
- Collection methodology and sampling logic
- Known blind spots and bias risks
- Transformation lineage (cleaning, aggregation, overrides)

**Output:** Provenance confidence rating

---

### 4️⃣ Legal & Compliance Checks (Non-Negotiable Gate)

**Question:** *Can Kuinbee legally sell this dataset?*

This is a **hard gate**:
- Ownership and resale rights
- Sub-licensing permissions
- PII and re-identification risk
- DPDP Act and Indian IT Act alignment
- Source website TOS compliance (for scraped data)

⚠️ If legal checks fail, the dataset is **not sellable**, regardless of other scores.

**Output:** Legal clearance status (Green / Amber / Red)

---

### 5️⃣ Market & Buyer Usability Checks (Commercial Layer)

**Question:** *Will anyone actually buy and use this dataset?*

Includes:
- Buyer ICP match and industry use-cases
- Ready-to-query vs raw data
- Joinability with public datasets
- Update and refresh feasibility

**Output:** Commercial viability score

---

## 📊 KDTS – Kuinbee Data Trust Score

All stages roll up into a final score:

