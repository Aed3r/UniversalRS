Install required packages:
```
pip install -r requirements.txt
```

Create documentation tree:
```
sphinx-quickstart docs
```

Edit `docs/conf.py` and create HTML documentation:
```
docs/make.bat html
```

Run all unit tests:
```
python -m unittest discover -s tests -p test_*.py
```