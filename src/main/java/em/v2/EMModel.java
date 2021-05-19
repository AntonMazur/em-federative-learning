package em.v2;

import JSci.maths.matrices.AbstractDoubleMatrix;
import JSci.maths.matrices.AbstractDoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;
import org.apache.commons.math3.util.FastMath;

public class EMModel implements Cloneable{

    public EMStepData data;

    public int dim;
    public int nClusters;
    public Cluster[] clusters;
    public double[] nonNormClustersWeights;

    public double getClusterWeight(int index) {
        return clusters[index].weight;
    }

    public EMModel copyForWorker() {
        try {
            EMModel copy = (EMModel) super.clone();
            copy.data = data.clone();

            return copy;

        } catch(CloneNotSupportedException ex) {
            throw new RuntimeException(ex);
        }

    }

    public static class Cluster {
        private double weight;
        private AbstractDoubleVector mean;
        private AbstractDoubleSquareMatrix cov;
        private double covDet;
        private AbstractDoubleSquareMatrix covInv;

        private AbstractDoubleVector workerLocalMean;
        private AbstractDoubleSquareMatrix workerLocalCov;

        public void updateGaussDistParams(AbstractDoubleVector mean, AbstractDoubleSquareMatrix covariance) {
            this.mean = mean;
            this.cov = covariance;
            covDet = covariance.det();
            covInv = covariance.inverse();
        }

        public double computeProbability(AbstractDoubleVector point) {
            double prob = FastMath.pow(2 * FastMath.PI, -0.5 * point.dimension()) * FastMath.pow(covDet, -0.5);
            double mahaDist = point.scalarProduct(covInv.multiply(point));
            double expTerm = FastMath.pow(FastMath.E, -0.5 * mahaDist);
            return prob * expTerm;
        }
    }


    static abstract class EMStepData implements Cloneable {

        public EMStepData clone() {
            try {
                return (EMStepData) super.clone();
            } catch(CloneNotSupportedException ex) {
                throw new RuntimeException(ex);
            }

        }
    }

    public static class EStepData extends EMStepData {
        // input
        public double[] clustersWeights;

        //output
        public double[] nonNormLocalClustersWeights;
    }

    public static class MStepStage1Data extends EMStepData {
        // input

        // output
        public AbstractDoubleVector[] clustersNonNormLocalMeans;
    }

    public static class MStepStage2Data extends EMStepData {
        // input
        public AbstractDoubleVector[] newClustersMeans;

        // output
        public AbstractDoubleSquareMatrix[] clustersNonNormLocalCovs;
    }


}





