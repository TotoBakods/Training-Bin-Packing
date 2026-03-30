# Training and Validation Data Samples (All Algorithms)

This document provides snapshots of the raw data used for training and validating the four model variants. Each variant is trained on data labeled by a specific heuristic algorithm.

> [!IMPORTANT]
> All samples are extracted using a **20% validation split** with a fixed random seed of `42` to match the actual training pipeline configuration.

## Algorithm: EO
**Source File**: `fit_eo.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.99 |     0.54 |     0.53 |   23.0007  |    8.16385 |       0    |
|     1.07 |     0.54 |     0.49 |   21.0844  |   10.7054  |       0    |
|     0.81 |     0.34 |     0.42 |    0.17    |    1.245   |       0.62 |
|     0.66 |     0.41 |     0.61 |    9.73388 |    5.04921 |       0    |
|     1.08 |     0.51 |     0.53 |   13.9828  |    2.71035 |       0    |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.74 |     0.39 |     0.24 |   1.095    |    0.37    |       0.53 |
|     1.15 |     0.5  |     0.55 |   1.755    |    3.03    |       0.55 |
|     0.97 |     0.3  |     0.47 |   9.40094  |   11.743   |       0    |
|     1.07 |     0.58 |     0.22 |   0.79     |    1.535   |       0.4  |
|     1.14 |     0.43 |     0.44 |   0.583669 |    3.21853 |       0    |

---

## Algorithm: GA
**Source File**: `fit_ga.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     1.01 |     0.59 |     0.33 |    8.26929 |   12.5169  |       0    |
|     1.11 |     0.51 |     0.26 |    6.39934 |    4.46442 |       0    |
|     1.17 |     0.77 |     0.5  |    1.60896 |    3.15928 |       0.44 |
|     1.13 |     0.4  |     0.48 |    2.20628 |    7.80459 |       0    |
|     1    |     0.55 |     0.54 |   10.3769  |    8.56622 |       0    |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     1.16 |     0.39 |     0.52 |   2.16616  |   3.06019  |       0.46 |
|     0.7  |     0.52 |     0.43 |   0.978795 |   0.343094 |       0    |
|     0.78 |     0.53 |     0.59 |   6.9323   |   4.78053  |       0    |
|     0.58 |     0.42 |     0.55 |   1.29     |   1.71     |       0.78 |
|     1.12 |     0.46 |     0.4  |   4.37631  |   9.23176  |       0    |

---

## Algorithm: EO + GA
**Source File**: `fit_eo_ga.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.67 |     0.42 |     0.61 |   8.84624  |    8.54266 |       0    |
|     0.67 |     0.41 |     0.6  |   0.523653 |    4.69059 |       0    |
|     0.56 |     0.43 |     0.38 |   1.06543  |    2.81983 |       0.62 |
|     0.84 |     0.69 |     0.58 |   5.845    |    2.92    |       0    |
|     0.8  |     0.58 |     0.36 |  14.8744   |   14.7387  |       0    |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.65 |     0.41 |     0.59 |    1.50907 |   0.500896 |       1.08 |
|     1.12 |     0.38 |     0.49 |    4.14    |   1.27     |       0.94 |
|     0.6  |     0.42 |     0.5  |    6.06917 |   5.37928  |       0    |
|     1.19 |     0.78 |     0.45 |    1.71    |   1.365    |       0.53 |
|     0.67 |     0.43 |     0.61 |    1.71467 |  10.2      |       0    |

---

## Algorithm: GA + EO
**Source File**: `fit_ga_eo.csv`

### Training Samples (80%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0.69 |     0.43 |     0.64 |   0.385146 |   2.74567  |       0    |
|     0.57 |     0.42 |     0.43 |   1.38166  |   0.598422 |       0    |
|     0.79 |     0.49 |     0.59 |   2.275    |   0.745    |       1.05 |
|     0.82 |     0.36 |     0.34 |   1.91231  |   5.70605  |       0    |
|     1.01 |     0.31 |     0.47 |   5.37769  |  16.7594   |       0    |

### Validation Samples (20%)
|   item_l |   item_w |   item_h |   target_x |   target_y |   target_z |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     1.13 |     0.45 |     0.57 |   0.765    |    2.495   |       1.03 |
|     1.01 |     0.46 |     0.49 |   3.28757  |    1.10786 |       0.98 |
|     1.08 |     0.47 |     0.57 |   0.291726 |    7.95932 |       0    |
|     0.76 |     0.52 |     0.61 |   1.88     |    0.74    |       1.25 |
|     1.11 |     0.4  |     0.52 |  20.78     |    9.52007 |       0    |

---

# Independent Test Set (GAN-Generated Data)

The test set is structurally independent from the training data. These samples represent synthetic warehouse scenarios generated by the GAN to evaluate the model's final generalization capability.

## Test Dataset: 200 Items
**Source File**: `gan/200_items.csv`

|   length |   width |   height |   weight | category       |
|---------:|--------:|---------:|---------:|:---------------|
|     0.81 |    0.55 |     0.44 |     9.24 | poultry        |
|     0.73 |    0.58 |     0.58 |     8.04 | confectionery  |
|     0.67 |    0.41 |     0.61 |    10.38 | candy          |
|     1.12 |    0.76 |     0.57 |    16.03 | dairy products |
|     0.56 |    0.3  |     0.44 |     4.98 | ice cream      |

## Test Dataset: 400 Items
**Source File**: `gan/400_items.csv`

|   length |   width |   height |   weight | category      |
|---------:|--------:|---------:|---------:|:--------------|
|     1.15 |    0.41 |     0.47 |    14.21 | pizza         |
|     1.07 |    0.44 |     0.48 |     8.79 | pizza         |
|     1.14 |    0.43 |     0.51 |    17.9  | confectionery |
|     1.12 |    0.47 |     0.3  |    15.06 | instant meals |
|     0.78 |    0.54 |     0.58 |    19.76 | ice cream     |

## Test Dataset: 600 Items
**Source File**: `gan/600_items.csv`

|   length |   width |   height |   weight | category        |
|---------:|--------:|---------:|---------:|:----------------|
|     0.49 |    0.39 |     0.53 |     6.05 | confectionery   |
|     0.74 |    0.71 |     0.5  |    15.79 | bakery products |
|     1.17 |    0.77 |     0.56 |    14.67 | bakery products |
|     1.05 |    0.63 |     0.59 |     8.45 | instant meals   |
|     0.68 |    0.41 |     0.61 |    11.35 | vegetables      |
