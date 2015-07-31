package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

public class IRMI extends Filter implements SupervisedFilter {
	
	/**
	 * Class to make it possible to sort by index (by grabbing indices inside the Pair objects)
	 * http://stackoverflow.com/questions/23587314/how-to-sort-an-array-and-keep-track-of-the-index-in-java
	 * @author cjb60
	 */
	public class Pair implements Comparable<Pair> {
		public final int value;
		public final int index;
		public Pair(int value, int index) {
			this.value = value;
			this.index = index;
		}
		@Override
		public int compareTo(Pair pair) {
			return -1 * Integer.valueOf(this.value).compareTo(pair.value);
		}
	}
	
	private static final long serialVersionUID = -6052407649001132182L;
	
	private Classifier m_nominalClassifier = new
			weka.classifiers.functions.Logistic();
	private Classifier m_numericClassifier = new
			weka.classifiers.functions.LinearRegression();
	
	private Classifier[] m_classifiers = null;
	
	private int m_numEpochs = 100;
	
	private double m_epsilon = 5;
	
	private boolean m_imputeTestData = true;
	
	public void setImputeTestData(boolean b) {
		m_imputeTestData = b;
	}
	
	public boolean getImputeTestData() {
		return m_imputeTestData;
	}
	
	public void setNominalClassifier(Classifier value) {
		m_nominalClassifier = value;
	}
	
	public Classifier getNominalClassifier() {
		return m_nominalClassifier;
	}
	
	public String nominalClassifierTipText() {
		return "Nominal classifier to use";
	}
	
	public void setNumericClassifier(Classifier value) {
		m_numericClassifier = value;
	}
	
	public Classifier getNumericClassifier() {
		return m_numericClassifier;
	}
	
	public String numericClassifierTipText() {
		return "Numeric classifier to use";
	}
	
	public void setNumEpochs(int x) {
		this.m_numEpochs = x;
	}
	
	public int getNumEpochs() {
		return this.m_numEpochs;
	}
	
	public String numEpochsTipText() {
		return "Max number of epochs";
	}
	
	public void setEpsilon(double x) {
		this.m_epsilon = x;
	}
	
	public double getEpsilon() {
		return this.m_epsilon;
	}
	
	public String epsilonTipText() {
		return "Epsilon for early termination";
	}
	
	public String globalInfo() {
		return "Iterative imputation";
	}
	
	public Capabilities getCapabilities() {
		
	    Capabilities result = super.getCapabilities();
	    result.disableAll();

	    // attributes
	    result.enableAllAttributes();
	    result.enable(Capability.MISSING_VALUES);
	    
	    // class
	    result.enableAllClasses();
	    result.enable(Capability.MISSING_CLASS_VALUES);
	    //result.enable(Capability.NO_CLASS);
	    
	    return result;
	}
	
	/**
	* Sets the format of the input instances.
	*
	* @param instanceInfo an Instances object containing the input 
	* instance structure (any instances contained in the object are 
	* ignored - only the structure is required).
	* @return true if the outputFormat may be collected immediately
	* @throws Exception if the input format can't be set 
	* successfully
	*/
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);
		setOutputFormat(instanceInfo);
		return true;
	}
	
	public static void main(String[] args) {
		 runFilter(new IRMI(), args);
	}
	
  /**
   * Input an instance for filtering. Filter requires all
   * training instances be read before producing output.
   *
   * @param instance the input instance
   * @return true if the filtered instance may now be
   * collected with output().
   * @throws IllegalStateException if no input format has been set.
   */
	public boolean input(Instance instance) throws Exception {
		if(m_Debug) {
			System.err.println("input()");
		}
		
		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}
		
		if ( isNewBatch() ) {
			if(m_Debug) System.err.println("  resetQueue()");
			resetQueue();
			m_NewBatch = false;
		}
		if ( isFirstBatchDone() == false ) {
			if(m_Debug) System.err.println("  bufferInput()");
			bufferInput(instance);
		} else {
			if(m_Debug) System.err.println("  convertInstance()");
			convertInstance(instance);
		}
		return isFirstBatchDone();
	}
	
	public void convertInstance(Instance inst) throws Exception {
		// push(inst); // do nothing
		
		if( getImputeTestData() ) {
			Instances instances = inst.dataset();
			int originalClassIndex = instances.classIndex();
			
			for(int x = 0; x < inst.numAttributes(); x++) {
				if( x == inst.classIndex() )
					continue;
				if(inst.isMissing(x)) {
					if( m_classifiers[x] != null ) {
						instances.setClassIndex(x);
						inst.setValue(x, m_classifiers[x].classifyInstance(inst) );
					}
				}
			}
			
			// not sure if this is needed
			instances.setClassIndex(originalClassIndex);
		}
		
		push(inst);
	}
	
	/*
	private double getProportionMissing(double[] data) {
		double nans = 0;
		for(int x = 0; x < data.length; x++) {
			if( Double.isNaN(data[x]) )
				nans += 1;
		}
		return nans / data.length;
	}
	*/
	
	/**
	 * NaN-friendly mean
	 * @param vals
	 * @return
	 */
	public double mean(double[] vals) {
		double sum = 0;
		double actualLength = 0;
		for(double val : vals) {
			if( Double.isNaN(val) == false ) {
				sum += val;
				actualLength += 1;
			}
		}
		sum = sum / actualLength;
		return sum;
	}
	
	public double median(double[] vals) {
		
		if(vals.length == 0) {
			return 0.0;
		}
		
		ArrayList<Double> newVals = new ArrayList<Double>(vals.length);
		for(double val : vals ) {
			if( Double.isNaN(val) == false ) {
				newVals.add(val);
			}
		}
		
		if(newVals.size() == 0) {
			return 0.0;
		}
		
		Collections.sort(newVals);
		
		// if the array is even, get the avg of the middle two
		int midPoint = newVals.size() / 2;
		if(newVals.size() % 2 == 0) {
			return (newVals.get(midPoint) + newVals.get(midPoint-1)) / 2.0;
		} else {
			return newVals.get(midPoint);
		}
	}
	
	public double mode(double[] vals) {
		
		if(vals.length == 0) {
			return 0.0;
		}
		
		Map<Double, Integer> counts = new HashMap<Double, Integer>();
		for(double num : vals) {
			if( Double.isNaN(num) ) {
				continue;
			}
			
			if( counts.get(num) == null) {
				counts.put(num, 1);
			} else {
				counts.put(num, counts.get(num) + 1);
			}
		}
		
		double bestKey = 0;
		double bestVal = Double.NEGATIVE_INFINITY;
		Set<Double> keys = counts.keySet();
		for(double key : keys) {
			if( counts.get(key) > bestVal ) {
				bestKey = key;
				bestVal = counts.get(key);
			}
		}
		
		return bestKey;
		
	}
	
	public int getNumMissing(double[] vals) {
		int sum = 0;
		for(double val : vals) {
			if(Double.isNaN(val))
				sum += 1;
		}
		return sum;
	}
	
	public void testknn(Instances df) throws Exception {
		
		//IBk knn = new IBk();
		//knn.buildClassifier(df);
		
	}
	
  /**
   * Signify that this batch of input to the filter is finished. 
   * If the filter requires all instances prior to filtering,
   * output() may now be called to retrieve the filtered instances.
   *
   * @return true if there are instances pending output
   * @throws IllegalStateException if no input structure has been defined
   */
	public boolean batchFinished() throws Exception 
	{	
		if(m_Debug) {
			System.err.println("batchFinished()");
		}
		
		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}
		
		if( isFirstBatchDone() == false ) {	
			System.err.println("  m_isDone == false");
			Instances df = getInputFormat();			
			int originalClassIndex = df.classIndex();
			
			/*
			 * Step 2: Sort the variables according to the original amount of
			 * missing values. We now assume that the variables are already sorted,
			 * i.e. M(x1) >= ... >= M(x2), where M(xj) denotes the number of missing
			 * cells in variable xj. Set I = {1..p}
			 */
			
			/*
			 * Step 4 prelim: Create a list of missing/observed indices
			 * with respect to each potential attribute.
			 */		
			Pair[] numMissing = new Pair[ df.numAttributes() ];
			ArrayList< ArrayList<Integer> > missingIndices = new ArrayList< ArrayList<Integer> >();
			ArrayList< ArrayList<Integer> > observedIndices = new ArrayList< ArrayList<Integer> >();
			for(int l = 0; l < df.numAttributes(); l++) {
				int missingCount = 0;
				ArrayList<Integer> missing = new ArrayList<Integer>();
				ArrayList<Integer> observed = new ArrayList<Integer>();
				for(int i = 0; i < df.numInstances(); i++) {
					Instance inst = df.get(i);
					if( inst.isMissing(l) ) {
						missing.add(i);
						missingCount += 1;
					} else {
						observed.add(i);
					}
				}
				missingIndices.add(missing);
				observedIndices.add(observed);
				numMissing[l] = new Pair(missingCount, l);
			}		
			Arrays.sort(numMissing);
			
			/*
			 * Step 1: Initialise the missing values using a simple imputation
			 * technique (e.g. k-nearest neighbour or mean imputation).
			 */
			for(int x = 0; x < df.numAttributes(); x++) {
				double[] vals = df.attributeToDoubleArray(x);
				
				double colMean = 0;	
				if(df.attribute(x).isNumeric()) {
					colMean = median(vals);
				} else { // it is nominal
					colMean = mode(vals);
				}
				
				for(int y = 0; y < vals.length; y++) {
					if( Double.isNaN( df.get(y).value(x) ) || 
							!Double.isFinite( df.get(y).value(x) ) ) {
						df.get(y).setValue(x, colMean);
					}
				}
			}			
			boolean[] isStable = new boolean[ df.numAttributes() ];
			for(int x = 0; x < df.numAttributes(); x++)
				isStable[x] = false;
			
			
			/*
			 * Step 4: Create a matrix with the variables corresponding to
			 * the observed and missing cells of x_l. 
			 */	
			m_classifiers = new Classifier[ df.numAttributes() ];
			
			
			/*
			 * Step 3: Set l = 0 (i.e. l = 1)
			 */	
			for(int epochs = 0; epochs < getNumEpochs(); epochs++) {
				if(m_Debug)
					System.out.println("Epoch #" + epochs);
				for(Pair p : numMissing) {
					int l = p.index;
					if(m_Debug)
						System.out.println(l);		
					if( p.value == 0 ) // none missing
						continue;
					if( l == originalClassIndex )
						continue;
					if( isStable[l] )
						continue;
					if( observedIndices.get(l).size() == 0 )
						continue;

					Instances observed = new Instances( getOutputFormat() );
					for(int x : observedIndices.get(l))
						observed.add( df.get(x) );
					observed.setClassIndex(l);
					
					Classifier cls = null;
					if(df.attribute(l).isNominal())
						cls = AbstractClassifier.makeCopy(m_nominalClassifier);
					else
						cls = AbstractClassifier.makeCopy(m_numericClassifier);
					
					/*
					 * Step 5: Build the model. Then predict the missing class.
					 */
					System.out.println(observed);
					cls.buildClassifier(observed);
					m_classifiers[l] = cls;
					
					
					double sumOfSquares = 0;
					df.setClassIndex(l);
					for(int idx : missingIndices.get(l)) {				
						double currentClassValue = df.get(idx).value(l);
						double newClassValue = m_classifiers[l].classifyInstance(df.get(idx));
						df.get(idx).setValue(l, newClassValue);
						sumOfSquares += Math.pow(currentClassValue - newClassValue, 2);
					}
					
					//ArffSaver as = new ArffSaver();
					//as.setInstances(df);
					//as.setFile(new File("/tmp/" + epochs + "_" + l + ".arff"));
					//as.writeBatch();
					
					if(sumOfSquares < m_epsilon) {
						isStable[l] = true;
						//System.out.println("Attr " + l + " is stable");
					}
					
					/*
					 * Step 6: Carry out steps 4-5 in turn for each l = 2..p.
					 */
				}

				boolean allStable = true;
				for(int j = 0; j < isStable.length; j++) {
					if(j == originalClassIndex) {
						continue;
					}
					else {
						if(isStable[j] == false) {
							allStable = false;
							break;
						}
					}			
				}
				if(allStable) {
					if(m_Debug) {
						System.out.println("All stable!");
					}
					break;
				}
				
			}
			for(Instance inst : df) {
				convertInstance(inst);
			}		
			//System.out.println("Done doing imputation");
		}
	    // Free memory
	    flushInput();
	    m_NewBatch = true;
	    m_FirstBatchDone = true;
	    if(m_Debug) System.err.println("numPendingOutput() = " + numPendingOutput());
	    return (numPendingOutput() != 0);
	}
	
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();
		
		result.addElement(new Option("Maximum number of epochs", "eh", 1, "-eh <max epochs>"));
		result.addElement(new Option("Epsilon", "en", 1, "-en <epsilon>"));
		result.addElement(new Option("Numeric classifier", "nc", 1, "-nc <classifier specification>"));
		result.addElement(new Option("Nominal classifier", "nl", 1, "-nl <classifier specification>"));
		
	    result.addAll(Collections.list(super.listOptions()));

	    return result.elements();	
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		
		// nl = nominal, nc = numeric
		
		String tmpStr = Utils.getOption("eh", options);
		if(tmpStr.length() != 0) {
			setNumEpochs( Integer.parseInt(tmpStr) );
		}
		
		tmpStr = Utils.getOption("en", options);
		if(tmpStr.length() != 0) {
			setEpsilon( Double.parseDouble(tmpStr) );
		}
		
		setImputeTestData(Utils.getFlag("it", options));
		
	    tmpStr = Utils.getOption("nl", options);
	    if (tmpStr.length() == 0) {
	    	tmpStr = weka.classifiers.functions.Logistic.class.getName();
	    }
	    String[] tmpOptions = Utils.splitOptions(tmpStr);
	    if (tmpOptions.length == 0) {
	      throw new Exception("Invalid classifier specification string");
	    }
	    tmpStr = tmpOptions[0];
	    tmpOptions[0] = "";
	    setNominalClassifier(AbstractClassifier.forName(tmpStr, tmpOptions));
	    
	    tmpStr = Utils.getOption("nc", options);
	    if (tmpStr.length() == 0) {
	    	tmpStr = weka.classifiers.functions.LinearRegression.class.getName();
		}
	    tmpOptions = Utils.splitOptions(tmpStr);
	    if (tmpOptions.length == 0) {
	      throw new Exception("Invalid classifier specification string");
	    }
	    tmpStr = tmpOptions[0];
	    tmpOptions[0] = "";
	    setNumericClassifier(AbstractClassifier.forName(tmpStr, tmpOptions));
	    
	    super.setOptions(options);

	    Utils.checkForRemainingOptions(options);
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		
		result.add("-eh");
		result.add( "" + getNumEpochs() );
		
		result.add("-en");
		result.add( "" + getEpsilon() );
		
		if( getImputeTestData() ) {
			result.add("-it");
		}
		
		result.add("-nl");
	    Classifier cls = getNominalClassifier();
	    String tmp = cls.getClass().getName();
	    if (cls instanceof OptionHandler) {
	    	tmp += " " + Utils.joinOptions(((OptionHandler) cls).getOptions());
	    }
	    result.add(tmp);
	    
		result.add("-nc");
	    cls = getNumericClassifier();
	    tmp = cls.getClass().getName();
	    if (cls instanceof OptionHandler) {
	    	tmp += " " + Utils.joinOptions(((OptionHandler) cls).getOptions());
	    }
	    result.add(tmp);
		
	    Collections.addAll(result, super.getOptions());
	    return result.toArray(new String[result.size()]);
	}

}
