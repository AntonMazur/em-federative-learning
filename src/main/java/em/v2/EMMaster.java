package em.v2;

import JSci.maths.matrices.AbstractDoubleSquareMatrix;
import JSci.maths.matrices.DoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class EMMaster {
    EMWorker[] workers;
    EMModel.GaussianMixture model;
    int nClusters;
    double[] clusterWeights;

    double previousLogLikehood = Double.NEGATIVE_INFINITY;

    public static class Config {
        public enum StopCriteria {
            NUM_ITERATIONS,
            LIKEHOOD_CHANGE
        }

        StopCriteria stopCriteria;

        int nMaxIter;
        double likehoodDelta;
        int nClusters;
    }

    public void iteration() {
        double[] clustProbs = Arrays.stream(workers)
                .map(client -> client.eStep(Optional.of(model), clusterWeights))
                .reduce((data1Probs, data2Probs) -> {
                            if (data1Probs.length == data2Probs.length && data1Probs.length == nClusters) {
                                double[] aggregatedProbs = new double[nClusters];

                                for (int i = 0; i < nClusters; i++) {
                                    aggregatedProbs[i] = data1Probs[i] + data2Probs[i];
                                }

                                return aggregatedProbs;
                            } else {
                                throw new RuntimeException("Clients return data from eStep() with different number of clusters");
                            }
                        }
                ).orElse(new double[0]);

        // aggregating of computations from each worker of means per them data and calculating general clusters' means
        AbstractDoubleVector[] nonNormClustersMeans = Arrays.stream(workers)
                .map(client -> client.mStepStage1(clustProbs))
                .reduce((data1Means, data2Means) -> {
                    if (data1Means.length == data2Means.length && data1Means.length == nClusters) {
                        AbstractDoubleVector[] aggregatedMeans = new AbstractDoubleVector[nClusters];

                        for (int i = 0; i < nClusters; i++) {
                            aggregatedMeans[i] = data1Means[i].add(data2Means[i]);
                        }

                        return aggregatedMeans;
                    } else {
                        throw new RuntimeException("Clients return data from mStepStage1() with different number of clusters");
                    }
                }).orElse(new AbstractDoubleVector[0]);

        AbstractDoubleVector[] clustersMeans = new AbstractDoubleVector[nClusters];

        for (int i = 0; i < nClusters; i++) {
            clustersMeans[i] = nonNormClustersMeans[i].scalarDivide(clustProbs[i]);
        }

        // aggregating of computations from each worker of covariance matrices per them data and calculating general clusters' covariances
        AbstractDoubleSquareMatrix[] nonNormClustersCov = Arrays.stream(workers)
                .map(client -> client.mStepStage2(clustersMeans))
                .reduce((data1Cov, data2Cov) -> {
                    if (data1Cov.length == data2Cov.length && data1Cov.length == nClusters) {
                        AbstractDoubleSquareMatrix[] aggregatedMeans = new DoubleSquareMatrix[nClusters];

                        for (int i = 0; i < nClusters; i++) {
                            aggregatedMeans[i] = data1Cov[i].add(data2Cov[i]);
                        }

                        return aggregatedMeans;
                    } else {
                        throw new RuntimeException("Clients return data from mStepStage1() with different number of clusters");
                    }
                }).orElse(new AbstractDoubleSquareMatrix[0]);

        AbstractDoubleSquareMatrix[] clustersCov = new DoubleSquareMatrix[nClusters];

        for (int i = 0; i < nClusters; i++) {
            clustersCov[i] = (DoubleSquareMatrix) nonNormClustersCov[i].scalarDivide(clustProbs[i]);
        }

        // updating covariance matrices for each worker
        Arrays.stream(workers).forEach(worker -> worker.mStepStage3(clustersCov));

    }

    public void run() {

        // todo: init params setup

        int iter = 0;
        while (iter++ < 500) {
            iteration();
        }

        // list of arrays in which each element denotes cluster index
        List<int[]> result = Arrays.stream(workers).map(EMWorker::getResult).collect(Collectors.toList());
    }


    public static void main(String[] args) {
        EMMaster server = new EMMaster();
        server.run();
    }

}
