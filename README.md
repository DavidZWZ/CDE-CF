### Install python environment
```bash
pip install -r requirements.txt  
```

## Datasets
We use the Amazon Review datasets. The data split is done in the leave-one-out setting for all users. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/).


```
python train.py --data_name=Beauty  --lr=0.001 --recdim=128 --solver='euler' --t=1 --decay=0.0001 --model_name=CDE_CF --epochs=1000
python train.py --data_name=Office_Products  --lr=0.001 --recdim=128 --solver='euler' --t=0.85 --decay=0.0001 --model_name=CDE_CF --epochs=1000
python train.py --data_name=Cell_Phones_and_Accessories  --lr=0.001 --recdim=128 --solver='euler' --t=0.9 --decay=0.0001 --model_name=CDE_CF --epochs=1000
python train.py --data_name=Health_and_Personal_Care  --lr=0.001 --recdim=128 --solver='euler' --t=0.65 --decay=0.0001 --model_name=CDE_CF --epochs=1000
```
