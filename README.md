# Wifi-Localization
ML models for Indoor Positioning System (IPS)
Author: Subhash karthik

*	Constructed ML models for Indoor positioning system to locate users using WIFI signal strength of their devices. Resulting model have a validation accuracy in the range 96%-98%
*	Classifiers like Naïve Bayes, Decision Trees, logistic regression was trained to illustrate interpretability

 This project aims to build a classifier for  identifying user's location using the Wi-Fi signal strength in their mobile devices. Actual System designs must take into account that at least three independent measurements are needed to unambiguously find a location. The system might include information from other systems to cope for physical ambiguity and to enable error compensation. Outliers in the data was allow removed to aid the system to build better models.
The first part of project does the data Cleaning, Preparation and used logistic regression as a classifier.
The second part employs Gaussian naïve Bayes and Decision trees as classifiers.

The resulting models have test accuracy in the range 96% - 98% .
The code is written in R markdown format with detailed description of the process employed.

 The first part project using logistic regression is explained at my blog( do checkout): https://subhash-karthik.github.io/Spam/
