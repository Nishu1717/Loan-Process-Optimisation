# Loan Process Optimisation Project

## Overview
This project focuses on optimizing the loan application process for a financial institution. It involves analyzing historical data to identify bottlenecks, predicting transition times, and proposing a "To-Be" process that automates incomplete file handling. Additionally, it includes a load balancing simulation to redistribute work from laggard users to efficient ones.

## Project Structure

```
Loan-Process-Optimisation/
├── data/               # Contains dataset files (CSV) - *Gitignored*
├── scripts/            # Python source code and analysis scripts
├── visualizations/     # Generated plots and citations
├── docs/               # Project documentation
├── notebooks/          # Jupyter notebooks for interactive analysis
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Loan-Process-Optimisation
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup**:
    - Place `bpi_2017_cleaned.csv` and `bpi_2017_hardened.csv` into the `data/` directory.
    - *Note: These files are not included in the repository due to size constraints.*
**Data Source**: https://www.kaggle.com/datasets/rascanudragos/bpi-challenge-2017

## Usage

### Running the Notebook
Navigate to the `notebooks/` directory and launch Jupyter Notebook:
```bash
cd notebooks
jupyter notebook "Loan_Process_Optimisation.ipynb"
```

### Running the Script
You can also run the analysis as a standalone Python script:
```bash
python scripts/loan_process_optimisation.py
```
This will generate visualizations in the `visualizations/` directory.

## Key Insights
- **Bottleneck Analysis**: Identified "Call incomplete files" as a major delay.
- **Process Automation**: Proposed AI-OCR integration to reduce incomplete file processing time by 90%.
- **Load Balancing**: Simulated a dynamic resource allocator that reduces queue time by redistributing tasks from the slowest 10% of users.

## Author
Nisarg Patel
