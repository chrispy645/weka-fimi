package weka.filters.unsupervised.attribute;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;

import org.junit.Test;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;

public class IRMITest {

	@Test
	public void testMedian() {
		
		IRMI x = new IRMI();
		
		double[] vals = new double[] { 1,5,8,3,1,2,4,6,3,2,1 };
		assertEquals( x.median(vals), 3.0, 0.0001);
		
		vals = new double[] { 1,5,8,3,1,2,4, Double.NaN, 6,3,2,1, Double.NaN };
		assertEquals( x.median(vals), 3.0, 0.0001);
		
		vals = new double[] { 1, 4, 5, 7 };
		assertEquals( x.median(vals), 4.5, 0.0001 );
		
		vals = new double[] { 1, 2 };
		assertEquals( x.median(vals), 1.5, 0.0001 );
		
		vals = new double[] { 1 };
		assertEquals( x.median(vals), 1.0, 0.0001 );
		
		vals = new double[] { };
		assertEquals( x.median(vals), 0.0, 0.0001 );
		
		vals = new double[] { Double.NaN, Double.NaN };
		assertEquals( x.median(vals), 0.0, 0.0001 );
		
	}
	
	@Test
	public void testMode() {
		
		IRMI x = new IRMI();
		
		double[] vals = new double[] { 1,1,1, 2,2, 3,3, 4,4,4,4 };
		assertEquals(x.mode(vals), 4.0, 0.0001);
		
		vals = new double[] { 1,1, Double.NaN, 1, 2,2, 3,3, 4,4,4,4, Double.NaN };
		assertEquals(x.mode(vals), 4.0, 0.0001);
		
		vals = new double[] { 1 };
		assertEquals(x.mode(vals), 1.0, 0.0001);
		
		vals = new double[] { };
		assertEquals( x.mode(vals), 0.0, 0.0001 );
		
		vals = new double[] { Double.NaN, Double.NaN };
		System.out.println(x.mode(vals));
		
	}
	
	@Test
	public void testImputation() throws Exception {
		
		String arff = "@relation test\n"
				+ "@attribute x1 numeric\n"
				+ "@attribute x2 {yes, no}\n"
				+ "@attribute class {c1, c2}\n"
				+ "@data\n"
				+ "1,yes,c1\n"
				+ "?,yes,c2\n"
				+ "2,?,c1\n"
				+ "3,no,c2\n";
		
		Instances df = DataSource.read(new ByteArrayInputStream(arff.getBytes()));
		df.setClassIndex(2);
		
		IRMI irmi = new IRMI();
		irmi.setNumEpochs(1000);
		irmi.setInputFormat(df);
		Instances newDf = Filter.useFilter(df, irmi);
		
		System.out.println(newDf.toString());
		
		// @data
		// 1,yes,c1
		// 2,yes,c2
		// 2,yes,c1
		// 3,no,c2
		
	}
	
}
