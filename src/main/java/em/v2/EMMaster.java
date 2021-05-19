package em.v2;

import JSci.maths.matrices.AbstractDoubleSquareMatrix;
import JSci.maths.matrices.DoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;
import JSci.maths.vectors.DoubleVector;

import javax.rmi.CORBA.Util;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class EMMaster {
    EMWorker[] workers;
    EMModel model;
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
        double[] aggregatedProbs = new double[nClusters];

        Arrays.stream(workers)
                .map(client -> client.eStep_(model.copyForWorker()))
                .forEach((newModel) -> {
                            double[] data1Probs = ((EMModel.EStepData) newModel.data).clustersWeights;

                            if (data1Probs.length == nClusters) {

                                for (int i = 0; i < nClusters; i++) {
                                    aggregatedProbs[i] += data1Probs[i];
                                }

                            } else {
                                throw new RuntimeException("Clients return data from eStep() with different number of clusters");
                            }
                        }
                );

        EMModel.MStepStage1Data newData = new EMModel.MStepStage1Data();
        model.data = newData;

        // aggregating of computations from each worker of means per them data and calculating general clusters' means

        AbstractDoubleVector[] aggregatedMeans = new AbstractDoubleVector[nClusters];
        Utils.init(aggregatedMeans, () -> new DoubleVector(model.dim));

        Arrays.stream(workers)
                .map(client -> client.mStepStage1_(model.copyForWorker()))
                .forEach(newModel -> {
                    AbstractDoubleVector[] clustersNonNormLocalMeans = ((EMModel.MStepStage1Data) newModel.data)
                            .clustersNonNormLocalMeans;

                    if (clustersNonNormLocalMeans.length == nClusters) {

                        for (int i = 0; i < nClusters; i++) {
                            aggregatedMeans[i] = aggregatedMeans[i].add(clustersNonNormLocalMeans[i]);
                        }

                    } else {
                        throw new RuntimeException("Clients return data from mStepStage1() with different number of clusters");
                    }
                });


        for (int i = 0; i < nClusters; i++) {
            aggregatedMeans[i] = aggregatedMeans[i].scalarDivide(aggregatedProbs[i]);
        }

        EMModel.MStepStage2Data dataMStepStage2 = new EMModel.MStepStage2Data();
        dataMStepStage2.newClustersMeans = aggregatedMeans;
        model.data = newData;

        // aggregating of computations from each worker of covariance matrices per them data and calculating general clusters' covariances
        AbstractDoubleSquareMatrix[] aggregatedCovs = new DoubleSquareMatrix[nClusters];
        Utils.init(aggregatedCovs, () -> new DoubleSquareMatrix(model.dim));

        Arrays.stream(workers)
                .map(client -> client.mStepStage2_(model.copyForWorker()))
                .forEach((newModel) -> {

                    AbstractDoubleSquareMatrix[] clustersNonNormLocalCovs = ((EMModel.MStepStage2Data) newModel.data)
                            .clustersNonNormLocalCovs;
                    if (clustersNonNormLocalCovs.length == nClusters) {

                        for (int i = 0; i < nClusters; i++) {
                            aggregatedCovs[i] = aggregatedCovs[i].add(clustersNonNormLocalCovs[i]);
                        }

                    } else {
                        throw new RuntimeException("Clients return data from mStepStage1() with different number of clusters");
                    }
                });


        for (int i = 0; i < nClusters; i++) {
            aggregatedCovs[i] = (DoubleSquareMatrix) aggregatedCovs[i].scalarDivide(aggregatedProbs[i]);
        }

        for (int i = 0; i < nClusters; i++) {
            model.clusters[i].updateGaussDistParams(aggregatedMeans[i], aggregatedCovs[i]);
        }

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
