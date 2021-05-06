package em;

import java.util.Arrays;
import java.util.Hashtable;
import java.util.Vector;

import JSci.maths.matrices.AbstractDoubleMatrix;
import JSci.maths.matrices.DoubleDiagonalMatrix;
import JSci.maths.matrices.DoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;
import JSci.maths.vectors.DoubleSparseVector;
import JSci.maths.vectors.DoubleVector;

public class Model {
    public static double accuracy = 1e-10;
    public static final int maxiter = 500;
    private static DoubleDiagonalMatrix stable; // for calc stability
    private boolean converged;
    private double llh;
    private double previousllh;
    private int count;
    private int dimension;
    private int vectorCount;
    private int clusterCount;
    //  label of assignment for each data entry, if converged
    private int[] finalAssignment;

    // clusters weights
    private double[] weight;

    // membership probability
    private double[][] memberProb;

    private AbstractDoubleVector[] data; // input data
    private AbstractDoubleVector[] mu; // mean or center
    private AbstractDoubleMatrix[] sigma;// covariance matrices

    public Model() {
        converged = false;
        count = 1;
    }

    private void initialization(AbstractDoubleVector[] data, int init) {
        int k = init;
        int n = vectorCount;
        mu = new DoubleVector[k];
        // idx is an array storing distinct k random values ranging from 1 to n
        int[] idx = RandomSample.randomsample(n, k);
        for (int i = 0; i < k; i++) {
            mu[i] = data[idx[i]];
        }

        double[][] temp = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                temp[i][j] = mu[j].scalarProduct(data[i]) - 0.5
                        * mu[j].scalarProduct(mu[j]);
            }
        }
        int[] label = new int[n];
        for (int i = 0; i < n; i++) {
            label[i] = max(temp[i]);
        }


        int uniqueelement = unique(label);
        while (k != uniqueelement) {
            idx = RandomSample.randomsample(n, k);
            for (int i = 0; i < k; i++) {
                mu[i] = data[idx[i]];
            }
            temp = new double[n][k];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < k; j++) {
                    temp[i][j] = mu[j].scalarProduct(data[i]) - 0.5
                            * mu[j].scalarProduct(mu[j]);
                }
            }
            label = new int[n];
            for (int i = 0; i < n; i++) {
                label[i] = max(temp[i]);
            }
            uniqueelement = unique(label);
        }
        /*
         * initialze the membership probability
         */
        memberProb = new double[n][k];
        for (int i = 0; i < n; i++) {
            memberProb[i][label[i]] = 1;
        }
    }

    /*
     * return the index of the maximum values in an array
     */
    private static int max(double[] array) {
        int idx = 0;
        double max = array[0];
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                idx = i;
            }
        }
        return idx;
    }

    /*
     * return the number of unique elements in an array. For example, if arr =
     * {1,2,3,4}, unique(arr) = 4; if arr = {1,1,2}, unique(arr) =2
     */
    private static int unique(int[] arr) {
        int ans = 0;
        Hashtable<Integer, Integer> ht = new Hashtable<Integer, Integer>();
        for (int i = 0; i < arr.length; i++) {
            if (ht.get(arr[i]) == null) {
                ht.put(arr[i], 1);
            } else {

            }
        }
        ans = ht.size();
        return ans;
    }

    public static void printArray(double[][] memberProb) {
        for (int i = 0; i < memberProb.length; i++) {
            for (int j = 0; j < memberProb[i].length; j++) {
                System.out.print(memberProb[i][j] + " ");
            }
            System.out.println("");
        }
    }

    private void maximization(double[][] memberProb, int n, int k, int d) {
        double[] temp = new double[k];
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < n; j++) {
                temp[i] += memberProb[j][i];
            }
        }
        // update weight or mixture portion
        for (int i = 0; i < k; i++) {
            weight[i] = temp[i] * 1.0 / n;
        }
        // update mean or center
        for (int h = 0; h < k; h++) {
            mu[h] = new DoubleVector(d);
            for (int i = 0; i < n; i++) {
                mu[h] = mu[h].add(data[i].scalarMultiply(memberProb[i][h]));
            }
            mu[h] = mu[h].scalarMultiply(1.0 / temp[h]);
        }
        // update sigma
        sigma = new DoubleSquareMatrix[k];
        for (int i = 0; i < k; i++) {
            sigma[i] = new DoubleSquareMatrix(d);
        }
        for (int h = 0; h < k; h++) {
            for (int i = 0; i < n; i++) {
                AbstractDoubleVector tempVector = (data[i].subtract(mu[h]))
                        .scalarMultiply(Math.sqrt(memberProb[i][h]));
                sigma[h] = (DoubleSquareMatrix) sigma[h].add((vectorSquare(tempVector)));
            }
            sigma[h] = sigma[h].scalarDivide(temp[h]);

            sigma[h] = sigma[h].add(stable); // for numerical stability
            // System.out.print(sigma[h]);
        }

    }

    private AbstractDoubleMatrix vectorSquare(AbstractDoubleVector vector) {
        int dim = vector.dimension();
        double[][] result = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                result[i][j] = vector.getComponent(i) * vector.getComponent(j);
            }
        }
        return new DoubleSquareMatrix(result);
    }

    private double expectation(int n, int k, int d) {
        double llh = 0;
        double[][] temp = new double[n][k];
        for (int h = 0; h < k; h++) {

            AbstractDoubleMatrix U = ((DoubleSquareMatrix) sigma[h])
                    .choleskyDecompose()[0];
            AbstractDoubleMatrix inverse = ((DoubleSquareMatrix) U).inverse();
            double t = 0;
            for (int i = 0; i < d; i++) {
                t += Math.log(U.getElement(i, i));
            }
            double c = d * Math.log(2 * Math.PI) + 2 * t;
            for (int i = 0; i < n; i++) {
                AbstractDoubleVector diff = data[i].subtract(mu[h]);
                AbstractDoubleVector Q = inverse.multiply(diff);
                double q = Q.scalarProduct(Q); // Maha distance
                temp[i][h] = -0.5 * (c + q);
            }
        }

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                temp[j][i] += Math.log(weight[i]);
            }
        }

        double[] T = logsumexp(temp);

        for (int i = 0; i < T.length; i++) {
            llh += T[i];
        }
        llh = llh * 1.0 / n;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                temp[i][j] = temp[i][j] - T[i];
                temp[i][j] = Math.exp(temp[i][j]);
            }
        }
        // update membership probability
        memberProb = temp;
        return llh;
    }

    // Compute log sum while avoiding numerical underflow.
    private double[] logsumexp(double[][] arr) {
        double[] ans = new double[arr.length];
        double[] y = new double[arr.length];
        double[][] temp = new double[arr.length][arr[0].length];
        for (int i = 0; i < temp.length; i++) {
            temp[i] = Arrays.copyOf(arr[i], arr[i].length);
        }
        /*
         * subtract the maximum of each row, at the end add it back
         */
        for (int i = 0; i < temp.length; i++) {
            y[i] = max(temp[i]);
            for (int j = 0; j < temp[i].length; j++) {
                temp[i][j] -= y[i];
            }
        }
        for (int i = 0; i < temp.length; i++) {
            double sum = 0;
            for (int j = 0; j < temp[i].length; j++) {
                sum += Math.exp(temp[i][j]);
            }
            ans[i] = y[i] + Math.log(sum);
        }
        Vector<Integer> idx = new Vector<Integer>();
        for (int i = 0; i < y.length; i++) {
            if (Double.isInfinite(y[i]) || Double.isNaN(y[i])) {
                idx.add(i);
            }
        }
        if (idx.size() != 0) {
            for (int i = 0; i < idx.size(); i++) {
                ans[idx.get(i)] = y[idx.get(i)];
            }
        }
        return ans;
    }

    /*
     * data matrix is an n x d matrix, where d is the dimension of each vector,
     * n is the number of vectors, k : number of clusters , for example,
     * {{1,2},{ 3,4},{5,6}} represents three two-dimensional points
     */
    public void bulidCluster(double[][] array, int k) {
        vectorCount = array.length;
        dimension = array[0].length;
        // for numerical stability

        double[] arr = new double[dimension];
        for (int i = 0; i < dimension; i++)
            arr[i] = 1e-6;
        stable = new DoubleDiagonalMatrix(arr);
        clusterCount = k;
        data = new DoubleVector[vectorCount];
        weight = new double[k];

        System.out.println("Running ... ");

        for (int i = 0; i < array.length; i++) {
            double[] temp = new double[dimension];
            temp = Arrays.copyOf(array[i], array[i].length);
            boolean isSparse = false;
            int zerocount = 0;
            for (int h = 0; h < temp.length; h++) {
                if (temp[h] == 0) {
                    zerocount++;
                }
            }
            isSparse = zerocount > 0.8 * temp.length;
            if (!isSparse)
                data[i] = new DoubleVector(temp);
            else
                data[i] = new DoubleSparseVector(temp);
        }
        initialization(data, clusterCount);

        previousllh = Double.NEGATIVE_INFINITY;
        while (!converged && count < Model.maxiter) {
            count++;
            maximization(memberProb, vectorCount, k, dimension);
            llh = expectation(vectorCount, k, dimension);
            // relative
            converged = llh - previousllh < accuracy * Math.abs(llh);
            previousllh = llh;
        }

        if (converged) {
            finalAssignment = new int[vectorCount];
            for (int i = 0; i < vectorCount; i++) {
                finalAssignment[i] = max(memberProb[i]);
            }
        } else {
            System.out.println("NOT converged is 500 steps");
        }
    }

    public double[][] getMembershipProbability() {
        return memberProb;
    }

    public double[] getWeight() {
        return weight;
    }

    public int[] getLabel() {
        return finalAssignment;
    }

    public boolean isConverged() {
        return converged;
    }

    public void setRelativeTolenranceLevel(double eps) {
        accuracy = eps;
    }
}