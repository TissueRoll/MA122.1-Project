import java.util.*;
import java.math.*;
import java.io.*;
public class NeuralNetwork{
	private Random rng;
	private ArrayList<Weights> weightList;
	private ArrayList<Neurons> nodes;
	private PrintWriter pw;
	public NeuralNetwork() throws IOException,FileNotFoundException{
		rng=new Random();
		weightList=new ArrayList<Weights>();
		nodes=new ArrayList<Neurons>();
		pw=new PrintWriter("output_nn.txt");
	}
	public double[][] transpose(double[][] m){
		double[][] t=new double[m[0].length][m.length];
		for(int i=0;i<t.length;i++) for(int j=0;j<m.length;j++) t[i][j]=m[j][i];
		return t;
	}
	public double[][] matrixSubtract(double[][] a,double[][] b){ //order of parameters matters
		if(a.length!=b.length||a[0].length!=b[0].length) return null;
		double[][] result=new double[a.length][a[0].length];
		for(int i=0;i<b.length;i++) for(int j=0;j<b[0].length;j++) result[i][j]=a[i][j]-b[i][j];
		return result;
	}
	public double[][] matrixMultiply(double[][] a,double[][] b){ //order of parameters matters
		if(a[0].length!=b.length) return null;
		int a_rows=a.length;
		int a_cols=a[0].length;
		int b_rows=b.length;
		int b_cols=b[0].length;
		double[][] result=new double[a_rows][b_cols];
		for(int i=0;i<a_rows;i++) for(int j=0;j<b_cols;j++) for(int k=0;k<a_cols;k++) result[i][j]+=a[i][k]*b[k][j];
		return result;
	}
	public double[][] gradientMultiply(double[][] a,double [][] b){
		if(a.length>1||b.length>1||a[0].length!=b[0].length) return null;
		double[][] result=new double[1][b[0].length];
		for(int i=0;i<a[0].length;i++) result[0][i]=a[0][i]*b[0][i];
		return result;
	}
	public double sigmoid(double x){
		return 1.0/(1+Math.exp(-x));
	}
	public double relu(double x){
		return Math.max(0,x);
	}
	public double sigmoidPrime(double x){
		return x*(1-x);
	}
	public double reluPrime(double x){
		if(x>0) return 1;
		return 0;
	}
	public void clearWeights(){
		weightList=new ArrayList<Weights>();
	}
	public Weights[] getWeights(){
		Weights[] w=new Weights[weightList.size()];
		for(int i=0;i<w.length;i++) w[i]=weightList.get(i);
		return w;
	}
	public void setWeights(Weights[] w){
		clearWeights();
		for(int i=0;i<w.length;i++) weightList.add(w[i]);
	}
	public double[] feedForward(double[][] input,double[][] output,int[] topology,boolean[] isSigmoid,Weights[] w) throws IOException,FileNotFoundException{ //return errors
		//input and output both have one row
		double[] error=new double[output[0].length];
		nodes.add(new Neurons(input[0]));
		double[][] hidden={{0}};
		if(w==null){ //random weights
			w=new Weights[topology.length-1];
			for(int i=0;i<w.length;i++){
				double[][] weights=new double[topology[i]][topology[i+1]];
				for(int j=0;j<weights.length;j++) for(int k=0;k<weights[0].length;k++) weights[j][k]=rng.nextDouble();
				w[i]=new Weights(weights);
			}
		}
		for(int i=0;i<w.length;i++){
			weightList.add(w[i]);
			double[][] weights=w[i].toMatrix();
			if(i==0) hidden=matrixMultiply(input,weights);
			else hidden=matrixMultiply(hidden,weights);
			if(isSigmoid[i]) for(int j=0;j<hidden[0].length;j++) hidden[0][j]=sigmoid(hidden[0][j]);
			else for(int j=0;j<hidden[0].length;j++) hidden[0][j]=relu(hidden[0][j]);
			nodes.add(new Neurons(hidden[0]));
			pw.println("LAYER VALUES:");
			for(int j=0;j<hidden[0].length;j++) pw.print(hidden[0][j]+" ");
			pw.println();
			pw.println("WEIGHTS:");
			for(int k=0;k<weights.length;k++){
				for(int j=0;j<weights[0].length;j++) pw.print(weights[k][j]+" ");
				pw.println();
			}
		}
		for(int i=0;i<error.length;i++) error[i]=Math.pow(output[0][i]-hidden[0][i],2);
		pw.flush();
		return error;
	}
	public Weights[] backProp(double[] error,boolean[] isSigmoid) throws IOException,FileNotFoundException{
		Weights[] newWeights=new Weights[weightList.size()]; int index=newWeights.length-1;
		Neurons last=nodes.get(nodes.size()-1);
		nodes.remove(nodes.size()-1);
		double[][] derivatives=new double[1][last.getLength()];
		if(isSigmoid[index]) for(int i=0;i<derivatives[0].length;i++) derivatives[0][i]=sigmoidPrime(last.get(i));
		else for(int i=0;i<derivatives.length;i++) derivatives[0][i]=reluPrime(last.get(i));
		pw.println("DERIVATIVES");
		for(int i=0;i<derivatives[0].length;i++) pw.print(derivatives[0][i]+" ");
		pw.println();
		double[][] gradients=new double[error.length][1];
		pw.println("Gradients:");
		for(int i=0;i<gradients.length;i++){
			gradients[i][0]=error[i]*derivatives[0][i];
			pw.print(gradients[i][0]+" ");
		}
		pw.println();
		//getting new weights between output and last hidden layer
		Neurons nextLayer=nodes.get(nodes.size()-1);
		double[][] delta=matrixMultiply(gradients,nextLayer.toRow());
		double[][] prev=weightList.get(weightList.size()-1).toMatrix();
		weightList.remove(weightList.size()-1);
		double[][] bago=matrixSubtract(prev,transpose(delta));
		newWeights[index--]=new Weights(bago);

		//getting new weights between the other layers
		gradients=transpose(gradients);
		while(!weightList.isEmpty()){
			last=nextLayer;
			nodes.remove(nodes.size()-1);
			derivatives=new double[1][last.getLength()];
			if(isSigmoid[index]) for(int i=0;i<derivatives[0].length;i++) derivatives[0][i]=sigmoidPrime(last.get(i));
			else for(int i=0;i<derivatives.length;i++) derivatives[0][i]=reluPrime(last.get(i));
			gradients=matrixMultiply(gradients,transpose(prev));
			gradients=gradientMultiply(gradients,derivatives);
			for(int i=0;i<gradients[0].length;i++) pw.print(gradients[0][i]+" ");
			pw.println();
			nextLayer=nodes.get(nodes.size()-1);
			double[][] z=nextLayer.toColumn();
			prev=weightList.get(weightList.size()-1).toMatrix();
			weightList.remove(weightList.size()-1);
			delta=matrixMultiply(z,gradients);
			bago=matrixSubtract(prev,delta);
			newWeights[index--]=new Weights(bago);
			
		}
		pw.flush();
		return newWeights;
	}
	public static void main(String[] args) throws IOException{
		NeuralNetwork nn=new NeuralNetwork();
		double[][] in={{0.2,0.1,0.3,0.5}};
		double[][] out={{1,0,0}};
		boolean[] act={true,true};
		int[] top={4,2,3};
		double[][] wait1={
			{0.15,0.14},
			{0.02,0.24},
			{0.62,0.2},
			{0.34,0.25}
		};
		double[][] wait2={
			{0.22,0.07,0.58},
			{0.59,0.55,0.77}
		};
		Weights[] bigat={new Weights(wait1),new Weights(wait2)};
		double[] wth=nn.feedForward(in,out,top,act,bigat);
		System.out.println(wth[0]);
		System.out.println(wth[1]);
		System.out.println(wth[2]);
		// for(int i=0;i<50;i++){
			// double error=0;
			// for(int j=0;j<wth.length;j++) error+=wth[j];
			// pw.println(error);
			// Weights[] anuna=nn.backProp(wth,act);
			// wth=nn.feedForward(in,out,top,act,anuna);
		// }
	}
}