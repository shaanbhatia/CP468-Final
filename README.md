# CP468-Final
Final Project for CP468

To use these files you will need the following python libraries:
- NumPy
- PyTorch (only for the ann.ipynb file)
- Sklearn
- Seaborn
- Pandas

To prep the data for the notebooks to work properly:
1. Download data-final.csv from [here](https://www.kaggle.com/datasets/tunguz/big-five-personality-test)
2. Open the dataset using excel and use text-to-columns to make it into a standard csv, or amend the pd.read_csv() call in the first cell of each notebook to include "delimiter='\t'".

Predictive Analysis Notebook Files
1. knn.ipynb - implementation of K-nearest-neighbours
2. dt.ipynb - implementation of decision trees
3. reg.ipynb - implementation of linear and logistic regression
4. ann.ipynb - implementation of artificial neural network
5. Country_Analysis.ipynb - implementation of statistical data
