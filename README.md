# ai-talents-league

# AI Talents League

This repository contains the code and resources for the AI Talents League competition challenges.

## Project Structure

```
.
├── ai-talents-league-round-1/
│   ├── First Challenge2.ipynb    # Main analysis notebook
│   ├── Prediction/              
│   ├── Prediction.csv           # Prediction results
│   ├── sample_submission.csv    # Sample submission format
│   ├── submission.csv          # Final submission file
│   ├── test.csv               # Test dataset
│   ├── train.csv             # Training dataset
│   └── outputs/
│       └── output_Lreg.csv   # Linear regression output
```

## Challenge Description

This project focuses on a machine learning prediction challenge. The main analysis includes:
- Data preprocessing and cleaning
- Feature engineering
- Model training and evaluation
- Prediction generation

## Data Processing

The analysis pipeline includes:
- Handling missing values
- Feature selection (removed X1 and X9 columns)
- Data transformation and normalization

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages (recommended to use a virtual environment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-talents-league.git
cd ai-talents-league
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install jupyter pandas numpy scikit-learn
```

### Usage

1. Open the Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `ai-talents-league-round-1/First Challenge2.ipynb`
3. Run the cells in sequence to reproduce the analysis

## Results

The predictions are saved in:
- `Prediction.csv`
- `outputs/output_Lreg.csv`

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.