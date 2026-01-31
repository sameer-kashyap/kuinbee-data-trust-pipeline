
# Kuinbee Data Trust Pipeline (MVP)

A Python-based **data trust and validation pipeline** designed to evaluate datasets before they are listed on the Kuinbee marketplace.

This project converts Kuinbee’s **manual dataset credibility framework** into a **code-backed, repeatable workflow**, focusing on data quality, consistency, trust, and usability.

---

## 📌 Why this project exists

Most data marketplaces focus on **distribution and access**.  
Kuinbee focuses on **trust**.

Before any dataset is listed, we want to answer:
- Is the dataset technically usable?
- Does it make logical sense?
- Can it be trusted?
- Is it legally safe to sell?
- Does it actually have buyer demand?

This repository is an **MVP pipeline** that automates the *technical and consistency checks* while keeping **legal and provenance checks human-reviewed**.

---

## 🧠 What this pipeline does

The pipeline evaluates datasets in structured stages:

### 1️⃣ Parametric & Structural Checks
- Schema integrity
- Data type consistency
- Completeness
- Uniqueness

### 2️⃣ Statistical Sanity
- Range checks
- Impossible values
- Distribution anomalies
- Zero-variance columns

### 3️⃣ Consistency & Temporal Checks
- Cross-column logic
- Referential integrity
- Units consistency
- Temporal validity (or explicit NA handling)

### 4️⃣ Geographic Consistency (when applicable)
- Administrative boundary validation
- Geo checks applied only to spatial datasets
