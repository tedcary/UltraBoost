package weka.classifiers.meta;

/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *   
 *   This is research code, not production code.
 *
 */

/*
 *    ULTRABOOST.java
 *
 */

import java.util.ArrayList;

import java.util.Random;

import weka.classifiers.Classifier;

import weka.classifiers.RandomizableMultipleClassifiersCombiner;

import weka.core.*;

// theEvaluation class below is used to calculate the error
// in the original paper, the Ultraboost::calculateMSE method was used
// but that was binary classification
// the Evaluation class error calculation should generalize to multi-class nominal data
// and allows for different objective functions

import weka.classifiers.evaluation.Evaluation;

import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * <!-- globalinfo-start --> UltraBoost adaptively boosts heterogeneous
 * classifiers.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * A. F. Moustafa et al., “Color Doppler Ultrasound Improves Machine Learning
 * Diagnosis of Breast Cancer,” Diagnostics, 2020, doi:
 * 10.3390/diagnostics10090631. <br/>
 * S. S. Venkatesh, B. J. Levenback, L. R. Sultan, G. Bouzghar, and C. M.
 * Sehgal, “Going beyond a First Reader: A Machine Learning Methodology for
 * Optimizing Cost and Performance in Breast Ultrasound Diagnosis,” Ultrasound
 * Med. Biol., 2015, doi: 10.1016/j.ultrasmedbio.2015.07.020.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#article{moustafa_cary_sultan_schultz_conant_venkatesh_sehgal_2020, 
 *  title={Color Doppler Ultrasound Improves Machine Learning Diagnosis of Breast Cancer}, 
 *  volume={10}, 
 *  DOI={10.3390/diagnostics10090631}, 
 *  number={9}, 
 *  journal={Diagnostics}, 
 *  author={Moustafa, Afaf F. and Cary, Theodore W. and Sultan, Laith R. and Schultz, Susan M. and Conant, Emily F. and Venkatesh, Santosh S. and Sehgal, Chandra M.}, 
 *  year={2020}, 
 *  pages={631}
 *  }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#article{venkatesh2015going,
 * 	title={Going beyond a first reader: A machine learning methodology for optimizing cost and performance in breast ultrasound diagnosis},
 * 	author={Venkatesh, Santosh S and Levenback, Benjamin J and Sultan, Laith R and Bouzghar, Ghizlane and Sehgal, Chandra M},
 * 	journal={Ultrasound in medicine \& biology},
 * 	volume={41},
 * 	number={12},
 * 	pages={3148--3162},
 * 	year={2015},
 * 	publisher={Elsevier}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * 
 * <pre>
 *  -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)
 * </pre>
 * 
 * 
 * <pre>
 *  -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 *
 * @author Ted Cary (ted.cary@pennmedicine.upenn.edu)
 * @version $Revision: 1.0 $
 */
public class UltraBoost extends RandomizableMultipleClassifiersCombiner implements TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = 1212197208062019L;

	protected ArrayList<Double> m_WeightsClassifiers = new ArrayList<Double>();

	public UltraBoost() throws Exception {

		FilteredClassifier Filter0 = new FilteredClassifier();
		FilteredClassifier Filter1 = new FilteredClassifier();

		// Filter0.setClassifier(new weka.classifiers.bayes.NaiveBayes());
		// Filter1.setClassifier(new weka.classifiers.functions.Logistic());

		String[] options0 = weka.core.Utils.splitOptions(
				"-F \"weka.filters.unsupervised.attribute.RemoveType -V -T nominal\" -S 1 -W weka.classifiers.bayes.NaiveBayes");

		Filter0.setOptions(options0);

		String[] options1 = weka.core.Utils.splitOptions(
				"-F \"weka.filters.unsupervised.attribute.RemoveType -V -T numeric\" -S 1 -W weka.classifiers.functions.Logistic");

		Filter1.setOptions(options1);

		// make sure constituent classifiers get passed the seed, if any
		// you also could have used the -S switch in the args to set the seed to m_Seed
		// or to anything other than 1 above, then used SetOptions)

		Filter0.setSeed(m_Seed);
		Filter1.setSeed(m_Seed);

		Classifier[] Classifiers = { Filter0, Filter1 };

		this.m_Classifiers = Classifiers; // overwrites ZeroR that's set in super MultipleClassifiersCombiner

	}

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter
	 *         gui
	 */
	public String globalInfo() {

		TechnicalInformation paper_old;

		paper_old = new TechnicalInformation(Type.ARTICLE);
		paper_old.setValue(Field.AUTHOR, "Santosh S. Venkatesh");
		paper_old.setValue(Field.YEAR, "2015");
		paper_old.setValue(Field.TITLE, "Going beyond a First Reader: "
				+ "A Machine Learning Methodology for Optimizing Cost and Performance in Breast Ultrasound Diagnosis");
		paper_old.setValue(Field.JOURNAL, "Ultrasound Med Bio");
		paper_old.setValue(Field.VOLUME, "41");
		paper_old.setValue(Field.NUMBER, "12");
		paper_old.setValue(Field.PAGES, "3148-62");
		paper_old.setValue(Field.PUBLISHER, "World Federation for Ultrasound in Medicine & Biology");

		return "UltraBoost adaptively boosts (AdaBoosts) heterogeneous classifiers:"
				+ " a different classifier can be boosted at each stage."
				+ " Unlike canonical AdaBoost, classifiers are not assumed to be weak."
				+ " Typical use is to only configure two or three stages, each with a different classifier."
				+ " It is named UltraBoost because it was developed at the Ultrasound Research Lab at the University of Pennsylvania."
				+ "\n\n"
				+ "Tip: Each classifier can be passed a subsampling of features filtered to the classifier's natural attribute type."
				+ " This acts as a \"practitioner's regularization\" to reduce dimensionality."
				+ " For instance, naive Bayes naturally takes nominal attributes, whereas logistic regression takes numeric attributes."
				+ " In the default UltraBoost configuration, naive Bayes is the first classifier, regulated to only work on nominal types."
				+ " This is done by wrapping the base classifier with FilteredClassifier set with the RemoveType filter."
				+ " Study the default configuration to see how to regularize classifiers by filtering for their natural type."
				+ "\n\n"
				+ "Prior to Weka 3.8.4, there was a bug in Weka's RemoveType filter when used with invertSelection."
				+ " This bug will throw the message: \"Problem evaluating classifier. Attribute names are not unique.\""
				+ " Though not as general, an easy workaround to the bug is to not use invertSelection in RemoveType filters."
				+ "\n\n"
				+ "In practice, often performance is best when the first classifier is the strongest, and only the first classifier is regularized by type."
				+ " Stringing together many heterogeneous strong classifiers may quickly overfit, and performance will decrease between boosts, depending on your problem."
				+ " The default naive Bayes then logistic regression sequence was purpose-built to work with ultrasound data from two different domains (image and clinician)."
				+ " A general strategy is to follow a strong regularized first classifier with \"sweeper\" boosted weak classifiers, like traditional AdaBoost or LogitBoost."
				+ "\n\n" + "Any classifier that outputs probabilities can be boosted,"
				+ " however by default UltraBoost uses naive Bayes and logistic regression as described above and in the papers referenced below."
				+ "\n\n" + "For more information, see:" + "\n\n" + getTechnicalInformation().toString() + "\n\n"
				+ paper_old.toString();

	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {

		// return the newer paper, since it is Open Access and references the original
		// anyway

		TechnicalInformation paper;
		paper = new TechnicalInformation(Type.ARTICLE);
		paper.setValue(Field.AUTHOR, "Afaf F. Moustafa");
		paper.setValue(Field.YEAR, "2020");
		paper.setValue(Field.TITLE, "Color Doppler Ultrasound Improves Machine Learning Diagnosis of Breast Cancer");
		paper.setValue(Field.JOURNAL, "Diagnostics");
		paper.setValue(Field.VOLUME, "10");
		paper.setValue(Field.NUMBER, "9");
		paper.setValue(Field.PAGES, "631");
		paper.setValue(Field.PUBLISHER, "MDPI AG");

		return paper;

//		TechnicalInformation result;
//
//		result = new TechnicalInformation(Type.ARTICLE);
//		result.setValue(Field.AUTHOR, "Santosh S. Venkatesh");
//		result.setValue(Field.YEAR, "2015");
//		result.setValue(Field.TITLE, "Going beyond a First Reader:"
//				+ "A Machine Learning Methodology for Optimizing Cost and Performance in Breast Ultrasound Diagnosis.");
//		result.setValue(Field.JOURNAL, "Ultrasound Med Bio");
//		result.setValue(Field.NUMBER, "12");		
//		result.setValue(Field.VOLUME, "41");
//		result.setValue(Field.PAGES, "3148-62");
//		result.setValue(Field.PUBLISHER, "World Federation for Ultrasound in Medicine & Biology");
//
//		return result;
	}

	/**
	 * Returns combined capabilities of the base classifiers, i.e., the capabilities
	 * all of them have in common.
	 *
	 * @return the capabilities of the base classifiers
	 */
	public Capabilities getCapabilities() {
		Capabilities result;

		result = super.getCapabilities();
		result.setMinimumNumberInstances(10); // just arbitrarily say 10

		return result;
	}

	/**
	 * Calculates mean squared error. Works for binary classification. This method
	 * is just here for legacy reasons. The Evaluation class offers a more general
	 * way to find error.
	 * 
	 * @param classifier
	 * @param data
	 * @return
	 * @throws Exception if probaility distributions can't be retrieved
	 */
	public double calculateMSE(Classifier classifier, Instances data) throws Exception {

		double sumSqErr = 0.0d;

		for (Instance inst : data) {

			double actualClass = inst.classValue(); // should be 0 or 1
			double[] dist = classifier.distributionForInstance(inst);
			double weight = inst.weight();
			// double diff = 1.0d - dist[(int) actualClass]; // this works because p[0,1] =
			// dist[0,1]
			double diff = actualClass - dist[1]; // sign doesn't matter bc you square later
			// if this were not binary, you'd have to evaluate each of many classes
			// here, dist[0,1] are probabilities p0,p1 of classes 0,1
			// so:
			// if actualClass ac = 0: diff = 1 - p0 = 1 - dist[ac=0] = 1 - (1 - p1) =
			// -((ac=0) - p1)
			// if actualClass ac = 1: diff = 1 - p1 = 1 - dist[ac=1] = (ac=1) - p1
			double sqErr = weight * diff * diff;
			sumSqErr += sqErr;
		}

		// divide by sum of weights for mean
		// if all weights are 1, it's the same as numInstances

		return Math.sqrt(sumSqErr) / data.sumOfWeights();
	}

	/**
	 * Buildclassifier selects a classifier from the set of classifiers by
	 * minimizing error on the training data.
	 *
	 * @param data the training data to be used for generating the boosted
	 *             classifier.
	 * @throws Exception if the classifier could not be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		Instances newData = new Instances(data);
		newData.deleteWithMissingClass();

		Random random = new Random(m_Seed);
		newData.randomize(random);

		int numInstances = newData.numInstances();
		double weightSum = numInstances; // for normalization
		Classifier[] classifiers = getClassifiers();
		// Evaluation eval;
		double[] weights = new double[numInstances];

		// fill weights with weights from original data
		for (int i = 0; i < weights.length; i++) {
			weights[i] = newData.get(i).weight();
		}

		Evaluation eval = new Evaluation(newData);

		double classWeightSum = 0.0d;
		for (Classifier classifier : classifiers) {

			// normalize the weights and weight the instances
			for (int i = 0; i < newData.numInstances(); i++) {
				weights[i] /= weightSum;
				weights[i] *= 50 * classifiers.length; // should be at least 2 for two classes, etc
				if (classifier instanceof weka.core.WeightedInstancesHandler) {
					newData.get(i).setWeight(weights[i]);
				}
				;
			}

			classifier.buildClassifier(newData);

			// the old way to do calculate err was calculateMSE:
			// double err = calculateMSE(classifier, newData);

			// a more general way to do it is the Evaluation class:
			// (this could also let you use many different objective functions)
			// (but the papers used mean-squared-error, as below:)

			eval.evaluateModel(classifier, newData);

			double err = eval.rootMeanSquaredError();
			err *= err;

			double classWeight = (1.0d / classifiers.length) * Math.log((1.0 - err) / err);

			classWeightSum += classWeight;
			m_WeightsClassifiers.add(classWeight);

			weightSum = 0.0d;
			for (int i = 0; i < newData.numInstances(); i++) {
				Instance inst = newData.get(i);
				double actualClass = inst.classValue(); // should be 0 or 1
				double[] dist = classifier.distributionForInstance(inst);
				double diff = dist[1] - actualClass;
				weights[i] = weights[i] * Math.exp(classWeight * (diff * diff));
				weightSum += weights[i]; // will be used to normalize at beginning of next loop over classifiers
			}

		}

		// normalize the class weights
		for (int i = 0; i < m_WeightsClassifiers.size(); i++) {
			m_WeightsClassifiers.set(i, m_WeightsClassifiers.get(i) / classWeightSum);
		}

	}

	/**
	 * Returns class probabilities.
	 *
	 * @param instance the instance to be classified
	 * @return the distribution
	 * @throws Exception if instance could not be classified successfully
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {

		Classifier[] classifiers = getClassifiers();
		int numClasses = instance.numClasses();
		double[] dist = new double[numClasses];

		// Double[] dist = new Double[numClasses]; //{ 0.0d, 0.0d }; // should be
		// binary, so only two class probabilities

		for (int i = 0; i < classifiers.length; i++) {
			double[] dist_i = classifiers[i].distributionForInstance(instance);
			for (int j = 0; j < numClasses; j++) {

				dist[j] += dist_i[j] * m_WeightsClassifiers.get(i);
			}
		}

		return dist;

	}

	/**
	 * Output a representation of this classifier
	 * 
	 * @return a string representation of the classifier
	 */
	public String toString() {

		if (m_Classifiers.length == 0) {
			return "UltraBoost: No base schemes entered.";
		}

		String result = "UltraBoost\n\nBase classifiers\n\n";
		for (int i = 0; i < m_Classifiers.length; i++) {
			result += getClassifier(i).toString() + "\n\n";
		}

		return result;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1.0.0 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain the following arguments: -t training file [-T test
	 *             file] [-c class index]
	 * @throws Exception
	 */
	public static void main(String[] argv) throws Exception {
		runClassifier(new UltraBoost(), argv);
	}
}
