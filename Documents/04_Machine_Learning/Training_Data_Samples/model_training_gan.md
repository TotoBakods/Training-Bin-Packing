# Training and Validation Data Samples (All Algorithms)

This document provides snapshots of the raw data used for training and validating the four model variants. Each variant is trained on data labeled by a specific heuristic algorithm.

> [!IMPORTANT]
> All samples are extracted using a **20% validation split** with a fixed random seed of `42` to match the actual training pipeline configuration.

## Algorithm: EO
**Source File**: `fit_eo.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.79 |     0.56 |     0.52 |   10.78    |    5.395   |          0 |
|     0.59 |     0.43 |     0.45 |    6.87722 |    4.59361 |          0 |
|     0.76 |     0.46 |     0.34 |    4.06331 |   11.4955  |          0 |
|     0.78 |     0.53 |     0.51 |    2.00859 |    2.66075 |          0 |
|     0.78 |     0.53 |     0.53 |    2.3804  |    2.92028 |          0 |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     1.18 |     0.78 |     0.33 |    6.81756 |  15.6536   |       0    |
|     0.66 |     0.41 |     0.58 |    2.12925 |   0.946623 |       0.4  |
|     1.11 |     0.78 |     0.29 |    2.56649 |   0.713195 |       0    |
|     0.58 |     0.41 |     0.39 |    0.745   |   0.29     |       0.38 |
|     0.77 |     0.53 |     0.32 |    2.765   |   1.385    |       0    |

---

## Algorithm: EO + GA
**Source File**: `fit_eo_ga.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     1.15 |     0.38 |     0.4  |    1.575   |    1.87    |       0    |
|     0.76 |     0.34 |     0.35 |   10.9953  |   14.0669  |       0    |
|     0.77 |     0.51 |     0.7  |    1.525   |    2.885   |       0    |
|     0.58 |     0.42 |     0.45 |    2.34916 |    2.69819 |       0.28 |
|     0.8  |     0.39 |     0.55 |    0.4     |    1.695   |       0.33 |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.8  |     0.61 |     0.46 |   15.305   |    4.4     |       0    |
|     0.79 |     0.53 |     0.54 |    3.435   |    1.895   |       1.05 |
|     1.11 |     0.41 |     0.57 |    1.555   |    1.705   |       0    |
|     1.13 |     0.3  |     0.46 |    1.15    |    1.065   |       0.33 |
|     1.17 |     0.39 |     0.43 |    4.26893 |    8.70394 |       0    |

---

## Algorithm: GA
**Source File**: `fit_ga.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     1.09 |     0.56 |     0.2  |    4.84186 |   11.123   |       0    |
|     1.08 |     0.54 |     0.51 |   16.0756  |    7.9921  |       0    |
|     0.79 |     0.52 |     0.57 |    8.22347 |    3.9133  |       0    |
|     1.08 |     0.54 |     0.51 |    1.95979 |    0.62097 |       0    |
|     1.18 |     0.79 |     0.55 |    1.51915 |    2.44663 |       0.47 |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.78 |     0.53 |     0.51 |     9.0535 |   11.9024  |       0    |
|     0.65 |     0.41 |     0.58 |     3.185  |    1.325   |       0.36 |
|     0.78 |     0.53 |     0.54 |     1.39   |    1.135   |       1.01 |
|     0.79 |     0.39 |     0.55 |     0.195  |    1.895   |       0    |
|     0.78 |     0.33 |     0.33 |    17.5689 |    4.13914 |       0    |

---

## Algorithm: GA + EO
**Source File**: `fit_ga_eo.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.79 |     0.53 |     0.55 |    5.82293 |    6.39829 |       0    |
|     1.18 |     0.79 |     0.55 |   10.09    |    0.395   |       0    |
|     1.18 |     0.79 |     0.55 |   12.09    |    5.895   |       0    |
|     0.95 |     0.41 |     0.49 |    2.255   |    1.205   |       0.79 |
|     0.78 |     0.53 |     0.36 |    2.45    |    0.265   |       0    |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.78 |     0.6  |     0.35 |   10.7685  |    8.57653 |       0    |
|     1.18 |     0.8  |     0.22 |    1.59    |    0.9     |       0.99 |
|     0.77 |     0.34 |     0.35 |    1.53389 |    2.77421 |       0    |
|     0.52 |     0.39 |     0.53 |    2.485   |    0.26    |       0.79 |
|     0.76 |     0.52 |     0.64 |   15.5316  |    4.16756 |       0    |

---

# Independent Test Set (GAN-Generated Data)

The test set is structurally independent from the training data. These samples represent synthetic warehouse scenarios generated by the GAN to evaluate the model's final generalization capability.

## Test Dataset: 200 Items
**Source File**: `gan/200_items.csv`

|   length |   width |   height |   weight | category        |
|---------:|--------:|---------:|---------:|:----------------|
|     0.78 |    0.53 |     0.5  |    21.39 | bakery products |
|     1.11 |    0.46 |     0.47 |    14.76 | pizza           |
|     1.18 |    0.79 |     0.26 |    16.81 | candy           |
|     0.91 |    0.53 |     0.47 |    16.67 | side dish       |
|     1.18 |    0.79 |     0.55 |    12.58 | bakery products |

## Test Dataset: 400 Items
**Source File**: `gan/400_items.csv`

|   length |   width |   height |   weight | category        |
|---------:|--------:|---------:|---------:|:----------------|
|     1.17 |    0.79 |     0.51 |    14.29 | bakery products |
|     1.1  |    0.5  |     0.27 |    17.56 | confectionery   |
|     0.8  |    0.59 |     0.5  |    19.46 | confectionery   |
|     1.16 |    0.42 |     0.36 |    15.34 | confectionery   |
|     1.09 |    0.56 |     0.21 |    16.36 | ice cream       |

## Test Dataset: 600 Items
**Source File**: `gan/600_items.csv`

|   length |   width |   height |   weight | category      |
|---------:|--------:|---------:|---------:|:--------------|
|     0.77 |    0.57 |     0.38 |    10.59 | confectionery |
|     1.1  |    0.51 |     0.56 |     8.5  | vegetables    |
|     0.59 |    0.2  |     0.2  |     8.25 | snack         |
|     1.11 |    0.35 |     0.47 |     8.94 | candy         |
|     1.13 |    0.3  |     0.45 |    13.74 | confectionery |
