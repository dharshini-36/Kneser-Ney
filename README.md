# Search AutoComplete using 5-Gram Language Model

## Overview

This project implements a Search Query AutoComplete System using a Sparse High-Order 5-Gram Language Model with Kneser-Ney Smoothing from scratch.

No NLP libraries such as NLTK or KenLM are used.

## Features

- Sparse 5-gram storage
- Recursive Kneser-Ney smoothing
- Back-off probability calculation
- Autocomplete suggestions
- Sentence probability
- Streamlit web interface
- Model statistics dashboard

## Technologies

- Python
- Streamlit
- Pandas
- defaultdict

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The application will open automatically in your browser.

## Project Structure

```
Search-Autocomplete/
│
├── app.py
├── requirements.txt
└── README.md
```

## Assignment Requirements Covered

- ✔ 5-Gram Language Model
- ✔ Sparse Data Structures
- ✔ Kneser-Ney Smoothing
- ✔ Recursive Back-off
- ✔ Autocomplete Suggestions
- ✔ Probability Calculation
- ✔ Streamlit Web Application
