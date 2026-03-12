# Python interpreter
PYTHON = venv/bin/python

# Install dependencies
install:
	pip install -r requirements.txt

# Generate synthetic dataset
generate-data:
	venv/bin/python data/generate_retail_data.py
	
# Run EDA notebook
eda:
	jupyter notebook notebooks/

# Run the full pipeline
run: install generate-data

# Clean generated files
clean:
	rm -f data/retail_demand_dataset.csv