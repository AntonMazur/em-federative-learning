package em.v2;

import JSci.maths.matrices.AbstractDoubleMatrix;
import JSci.maths.matrices.AbstractDoubleSquareMatrix;
import JSci.maths.matrices.DoubleMatrix;
import JSci.maths.matrices.DoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;
import JSci.maths.vectors.DoubleVector;
import org.apache.commons.math3.util.FastMath;

import java.util.function.Supplier;

public class EMWorker {
    int nClusters;
    int dim;
    AbstractDoubleVector[] inputData;
    AbstractDoubleMatrix gamma; //hidden variables

    public EMWorker(AbstractDoubleVector[] inputData, int nClusters) {
        if (inputData.length == 0) {
            throw new RuntimeException("Client input data length must not be equal to zero!");
        } else {
            this.nClusters = nClusters;
            this.dim = inputData[0].dimension();
            this.inputData = inputData;
            this.gamma = new DoubleMatrix(inputData.length, inputData[0].dimension());
        }
    }


    public EMModel eStep_(EMModel model) {
        EMModel.EStepData data = (EMModel.EStepData) model.data;

        Supplier<EMModel> callableResult = () -> {
            AbstractDoubleVector[] elsProbs = new AbstractDoubleVector[inputData.length];
            AbstractDoubleVector elProb = new DoubleVector(nClusters);

            double[] result = new double[nClusters];

            for (int i = 0; i < inputData.length; i++) {
                int clustersProbSum = 0;
                EMModel.Cluster cluster = model.clusters[i];
                for (int j = 0; j < nClusters; j++) {
                    double prob = data.clustersWeights[j] * cluster.computeProbability(inputData[i]);
                    elProb.setComponent(j, prob);
                    clustersProbSum += prob;
                }

                for (int j = 0; j < nClusters; j++) {
                    gamma.setElement(i, j, elProb.getComponent(j) / clustersProbSum);

                    result[j] += gamma.getElement(i, j);
                }
            }

            data.nonNormLocalClustersWeights = result;

            return model;
        };

        return callableResult.get();
    }

    public EMModel mStepStage1_(EMModel model) {
        EMModel.MStepStage1Data data = (EMModel.MStepStage1Data) model.data;

        AbstractDoubleVector[] clustersLocalMeans = new AbstractDoubleVector[nClusters];
        Utils.init(clustersLocalMeans, () -> new DoubleVector(new double[dim]));

        for (int i = 0; i < inputData.length; i++) {
            for (int j = 0; j < nClusters; j++) {
                clustersLocalMeans[j].add(inputData[i].scalarMultiply(gamma.getElement(i, j)));
            }
        }

        data.clustersNonNormLocalMeans = clustersLocalMeans;
        return model;
    }

    public EMModel mStepStage2_(EMModel model) {
        EMModel.MStepStage2Data data = (EMModel.MStepStage2Data) model.data;


        AbstractDoubleSquareMatrix[] nonNormCovariance = new DoubleSquareMatrix[nClusters];
        Utils.init(nonNormCovariance, () -> new DoubleMatrix(dim, dim));

        for (int i = 0; i < inputData.length; i++) {
            for (int j = 0; j < nClusters; j++) {
                AbstractDoubleVector diff = inputData[i].subtract(data.newClustersMeans[j]);
                nonNormCovariance[j].add(Utils.vectorSquare(diff).scalarMultiply(gamma.getElement(i, j)));
            }
        }

        data.clustersNonNormLocalCovs = nonNormCovariance;
        return model;
    }


    public double getNonNormLogLikehood() {
        if (gamma == null) {
            throw new RuntimeException("Gamma is uninitialized!");
        }

        double logLikehood = 0;

        for (int i = 0; i < inputData.length; i++) {
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
            int[] dataAssignments = new int[inputData.length];

            for (int i = 0; i < inputData.length; i++) {
                dataAssignments[i] = Utils.indexOfMaxInRow(gamma, i);
            }

            return dataAssignments;
        }
    }
}
