Set up
```
python3 -m venv .env
pip install -r requirements.txt
```

Training, change --train_data_path parameter
```
python3 main.py --run_mode train --train_data_path '/ml_task/btcusd-h-30/data.h5'
```

Prediction, change --predict_data_path parameter
```
python3 main.py --run_mode predict --predict_data_path '/ml_task/btcusd-h-30/data.h5' --model_path 'model.data'
```

Requirements
```
Tested with resources: 2 CPU, 30GB memory.
Training duration takes about 20 min.
Inference takes about 5 minutes.
```
