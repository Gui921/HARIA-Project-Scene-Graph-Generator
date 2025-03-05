# SGG for the HARIA Project

## Data preperation
Just like the authors of the original repo, we used the dataset Action Genome to evaluate our framework. 
Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome) and put the processed annotation [files](https://drive.google.com/drive/folders/1tdfAyYm8GGXtO2okAoH1WgVHVOTl1QYe?usp=share_link) with COCO style into `annotations` folder. 
The directories of the dataset should look like:
```
|-- action-genome
    |-- annotations   # gt annotations
        |-- ag_train_coco_style.json
        |-- ag_test_coco_style.json
        |-- ...
    |-- frames        # sampled frames
    |-- videos        # original videos
```

## How to run

**Please download the [checkpoints](https://drive.google.com/drive/folders/12zh9ocGmbV8aOFPzUfp8ezP0pMTlpzJl?usp=sharing) used in the paper and put it into `/exps` folder.**

After downloading the checkpoints, one can run it simply by running the following command line:

`python run.py`

The script has the following arguments:

`--data_nr` Index of the data sample to process - default = 30

`--filter_rate` Filter rate threshold - default = 4.5

`--file` Image file path

## Model Performance:

| Task    | Module |W/R@10|W/R@20|W/R@50|N/R@10|N/R@20|N/R@50|weight|
|:-------:|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| SGDET   |spatial | 31.5 | 37.7 | 43.7 | 33.4 | 41.6 | 49.0 |[link](https://drive.google.com/drive/folders/1a1chPaZejEY3UrCf0zhTteBZio5wLNdS?usp=share_link)|
| SGDET   |temporal| 33.5 | 40.9 | 48.9 | 35.3 | 44.0 | 51.8 |[link](https://drive.google.com/drive/folders/1H_ldtbwe8f0maq_IieQBG6MmE3EXHbJV?usp=share_link)|
| PredCLS |spatial | 72.9 | 76.0 | 76.1 | 83.3 | 95.3 | 99.2 |[link](https://drive.google.com/drive/folders/1o-iMR_pSvJ0dqDcRlTgGal_hXpVfhosQ?usp=share_link)|
| PredCLS |temporal| 73.0 | 76.1 | 76.1 | 83.3 | 95.3 | 99.2 |[link](https://drive.google.com/drive/folders/1JhuHxzalRG_kVprM412jT8izWyDU622Y?usp=share_link)|
