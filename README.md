# UltraBoost
Weka UltraBoost package for adaptively boosting heterogeneous learners

UltraBoost will work best with Weka 3.8.4 or above, though it will work with reduced functionality on earlier releases. To install the package manually to Weka on a local machine, download the ultraBoost.zip file. From the Weka GUI Chooser menubar, select Tools->Package manager. Press the 'File/URL' button at the top right and navigate to the ultraBoost.zip file you downloaded, which will install the package. After installation, restart Weka. The UltraBoost classifier will now be available in the 'meta' classifiers. (In Weka Explorer, select the 'Classify' tab, press the 'Choose' button, and expand the tree to see weka->Classifiers->meta->UltraBoost.) 

NAME
weka.classifiers.meta.UltraBoost

SYNOPSIS
UltraBoost adaptively boosts (AdaBoosts) a series of heterogeneous classifiers. Unlike canonical AdaBoost, base classifiers are not assumed to be weak -- they can be relatively strong, and dissimilar. It is named UltraBoost because it was developed at the Ultrasound Research Lab at the University of Pennsylvania.

Bug: Prior to Weka 3.8.4, there was a bug in Weka's RemoveType filter when used with invertSelection. This bug will throw the message: "Problem evaluating classifier. Attribute names are not unique." Though not as general, an easy workaround to the bug is to not use invertSelection in RemoveType filters.

Tip: Each classifier can be passed a subsampling of features filtered to the classifier's natural attribute type. This acts as a "practitioner's regularization" to reduce dimensionality. For instance, naive Bayes naturally takes nominal attributes, whereas logistic regression takes numeric attributes. In the default UltraBoost configuration, naive Bayes is the first classifier, regulated to only work on nominal types. This is done by wrapping the base classifier with weka.classifiers.meta.FilteredClassifier with the RemoveType filter. Study the default configuration to see how to regularize classifiers by filtering their natural type.

In practice, often performance is best when the first classifier is the strongest, and only the first classifier is regularized by type. Stringing together many heterogeneous strong classifiers may quickly overfit, and performance will decrease between boosts, depending on your problem. The default naive Bayes then logistic regression sequence was purpose-built to work with ultrasound data from two different domains (image and clinician). A general strategy is to follow a strong regularized first classifier with "sweeper" boosted weak classifiers, like traditional AdaBoost or LogitBoost

Any base classifier that outputs probabilities can be boosted, however by default UltraBoost uses naive Bayes and logistic regression as described above and in the paper referenced below.

For more information, see:

Santosh S. Venkatesh (2015). Going beyond a First Reader:A Machine Learning Methodology for Optimizing Cost and Performance in Breast Ultrasound Diagnosis.. Ultrasound Med Bio. 41:3148-62.
