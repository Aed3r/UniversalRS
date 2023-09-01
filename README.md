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

Dataset:
 - Follow https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/MovieLens.md:
    - Download ml-100k.zip from http://files.grouplens.org/datasets/movielens/ml-100k.zip
    - Download ml-1m.zip from http://files.grouplens.org/datasets/movielens/ml-1m.zip
    - Unzip ml-100k.zip to `./data/ml-100k/`
    - Unzip ml-1m.zip to `./data/ml-1m/`
    - Create `./data/ml-100k/inter/` and `./data/ml-1m/inter/`
    - Convert `.data/ml-100k/u.item` and `.data/ml-1m/movies.dat` to UTF8 using notepad++
    - Run:
 ```
 python.exe .\RecDatasets\conversion_tools\run.py --dataset ml-100k --input_path .\data\ml-100k\ --output_path .\data\ml-100k\inter --convert_inter --convert_item --convert_user
python.exe .\RecDatasets\conversion_tools\run.py --dataset ml-1m --input_path .\data\ml-1m\ --output_path .\data\ml-1m\inter --convert_inter --convert_item --convert_user
 ```
 - Follow https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/MovieLens-KG.md:
    - Download Movielens-KG.zip from https://drive.google.com/drive/folders/1B5Bacbli0G9qW7Tm0H039eKww8rwKPUY
    - Unzip Movielens-KG.zip to ./data/Movielens-KG/
    - Create ./data/ml-100k/kg/ and ./data/ml-1m/kg/
    - Run:
    ```
    python .\RecDatasets\conversion_tools\add_knowledge.py --dataset ml-100k --inter_file ./data/ml-100k/inter/ml-100k.inter --kg_data_path ./data/MovieLens-KG --output_path ./data/ml-100k/kg --hop 1
    python .\RecDatasets\conversion_tools\add_knowledge.py --dataset ml-1m --inter_file ./data/ml-1m/inter/ml-1m.inter --kg_data_path ./data/MovieLens-KG --output_path ./data/ml-1m/kg --hop 1
    ```

Create a RSA key and put the id_rsa file in the root directory of the project:
http://www.linuxproblem.org/art_9.html
