#!/bin/bash

mkdir logs
mkdir output
mkdir images

# python filter-data-training.py -np 8 > logs/filter-data-training.log 2>&1


# python profile-dataset.py                   \
#     -nd     "Sage 2.0"      "output/sage-training-set.json" \
#     -nd     "Sage 2.3rc1"   "output/rc1-training-set.json"  \
#     -nd     "Sage 2.3rc2"   "output/training-set.json"  \
#     -o      "functional-group-profiles" > logs/profile-dataset.log 2>&1


# python plot-dataset-profile.py              \
#     -i  "functional-group-profiles/density-functional-groups.csv"   \
#     -o  "images/density-functional-groups.png"                      \
#     -s  "Density"                                                   \
#     -n 25 -w 8 -h 7

# python plot-dataset-profile.py              \
#     -i  "functional-group-profiles/dhmix-functional-groups.csv"   \
#     -o  "images/dhmix-functional-groups.png"                      \
#     -s  "∆Hmix"                                                   \
#     -n 25 -w 8 -h 7


# python compare-dataset-components.py \
#     -n "output/training-set.json" \
#     -o "output/sage-training-set.json" \
#     -f "functional-group-profiles/unique-components.json" \
#     -i "images" > logs/compare-dataset-components.log 2>&1


# python filter-data-validation.py -np 8 > logs/filter-data-validation.log 2>&1

# python profile-dataset.py                   \
#     -nd     "Sage 2.0"      "output/sage-training-set.json" \
#     -nd     "Sage 2.3rc2 training"   "output/training-set.json"  \
#     -nd     "Sage 2.3rc2 validation"   "output/validation-set.json"  \
#     -o      "functional-group-profiles-validation" > logs/profile-dataset-validation.log 2>&1


# python plot-dataset-profile.py              \
#     -i  "functional-group-profiles-validation/density-functional-groups.csv"   \
#     -o  "images/val-density-functional-groups.png"                      \
#     -s  "Density"                                                   \
#     -n 25 -w 8 -h 7

# python plot-dataset-profile.py              \
#     -i  "functional-group-profiles-validation/dhmix-functional-groups.csv"   \
#     -o  "images/val-dhmix-functional-groups.png"                      \
#     -s  "∆Hmix"                                                   \
#     -n 25 -w 8 -h 7


python filter-data-validation-extended.py            \
    -i ../intermediate/output/renamed-clean-filtered-viscosities.csv \
    -o output/validation-set-extended.csv   \
    -t output/training-set.json             \
    -t output/validation-set.json           \
    -np 8 > logs/filter-data-validation-extended.log 2>&1
