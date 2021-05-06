package em.v2;

import JSci.maths.matrices.AbstractDoubleMatrix;
import JSci.maths.matrices.AbstractDoubleSquareMatrix;
import JSci.maths.matrices.DoubleMatrix;
import JSci.maths.matrices.DoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;
import JSci.maths.vectors.DoubleVector;
import em.v2.EMModel.GaussianMixture;
import org.apache.commons.math3.util.FastMath;

import javax.rmi.CORBA.Util;
import java.util.Optional;
import java.util.function.Supplier;

public class EMWorker {
    int nClusters;
    int dim;
    AbstractDoubleVector[] data;
    AbstractDoubleMatrix gamma; //hidden variables
    GaussianMixture params;

    double[] nonNormWeights;
    AbstractDoubleVector[] newClustersMeans;


    public EMWorker(AbstractDoubleVector[] data, int nClusters) {
        if (data.length == 0) {
            throw new RuntimeException("Client input data length must not be equal to zero!");
        } else {
            this.nClusters = nClusters;
            this.dim = data[0].dimension();
            this.data = data;
            this.gamma = new DoubleMatrix(data.length, data[0].dimension());
        }
    }

    public double[] eStep(Optional<GaussianMixture> maybeModelParams, double[] clusterWeights) {
        maybeModelParams.ifPresent((modelParams) -> params = modelParams);

        Supplier<double[]> callableResult = () -> {
            AbstractDoubleVector[] elsProbs = new AbstractDoubleVector[data.length];
            AbstractDoubleVector elProb = new DoubleVector(nClusters);

            double[] result = new double[nClusters];

            for (int i = 0; i < data.length; i++) {
                int clustersProbSum = 0;
                GaussianMixture.Cluster cluster = params.clusters[i];
                for (int j = 0; j < nClusters; j++) {
                    double prob = clusterWeights[j] * cluster.computeProbability(data[i]);
                    elProb.setComponent(j, prob);
                    clustersProbSum += prob;
                }

                for (int j = 0; j < nClusters; j++) {
                    gamma.setElement(i, j, elProb.getComponent(j) / clustersProbSum);

                    result[j] += gamma.getElement(i, j);
                }
            }

            return result;
        };

        return callableResult.get();
    }

    // computes sums of probabilities for each data entry of belonging to a cluster
    public AbstractDoubleVector[] mStepStage1(double[] nonNormWeights) {
        this.nonNormWeights = nonNormWeights;

        AbstractDoubleVector[] clustersLocalMeans = new AbstractDoubleVector[nClusters];
        Utils.init(clustersLocalMeans, () -> new DoubleVector(new double[dim]));

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < nClusters; j++) {
                clustersLocalMeans[j].add(data[i].scalarMultiply(gamma.getElement(i, j)));
            }
        }

        return clustersLocalMeans;
    }

    public AbstractDoubleSquareMatrix[] mStepStage2(AbstractDoubleVector[] clustersMeans) {

        this.newClustersMeans = clustersMeans;

        AbstractDoubleSquareMatrix[] nonNormCovariance = new DoubleSquareMatrix[nClusters];
        Utils.init(nonNormCovariance, () -> new DoubleMatrix(dim, dim));

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < nClusters; j++) {
                AbstractDoubleVector diff = data[i].subtract(clustersMeans[j]);
                nonNormCovariance[j].add(Utils.vectorSquare(diff).scalarMultiply(gamma.getElement(i, j)));
            }
        }

        return nonNormCovariance;
    }

    public void mStepStage3(AbstractDoubleSquareMatrix[] clustersCov) {

        for (int i = 0; i < nClusters; i++) {
            params.clusters[i].updateGaussDistParams(newClustersMeans[i], clustersCov[i]);
        }
    }

    public double getNonNormLogLikehood() {
        if (gamma == null) {
            throw new RuntimeException("Gamma is uninitialized!");
        }

        double logLikehood = 0;

        for (int i = 0; i < data.length; i++) {
            double dataVectorLikehood = 0;

            for (int j = 0; j < nClusters; j++) {
                dataVectorLikehood += gamma.getElement(i, j);
            }

            logLikehood += FastMath.log(dataVectorLikehood);
        }

        return logLikehood;
    }

    public int[] getResult() {
        if (gamma == null) {
            throw new RuntimeException("Result is empty!");
        } else {
            int[] dataAssignments = new int[data.length];

            for (int i = 0; i < data.length; i++) {
                dataAssignments[i] = Utils.indexOfMaxInRow(gamma, i);
            }

            return dataAssignments;
        }
    }
}
