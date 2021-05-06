package em.v2;

import JSci.maths.matrices.AbstractDoubleMatrix;
import JSci.maths.matrices.AbstractDoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;
import org.apache.commons.math3.util.FastMath;

public class EMModel {

    public static class GaussianMixture {
        public int dim;
        public int nClusters;
        public Cluster[] clusters;
        public double[] clustersWeights;
        public AbstractDoubleVector[] clustersMeans; // clusterMeans[n][m], n - cluster num, m - dimension num
        public AbstractDoubleMatrix[] clustersCovariance;

        public static class Cluster {
            private double weight;
            private AbstractDoubleVector mean;
            private AbstractDoubleSquareMatrix covariance;
            private double covDet;
            private AbstractDoubleSquareMatrix covInv;

            public void updateGaussDistParams(AbstractDoubleVector mean, AbstractDoubleSquareMatrix covariance) {
                this.mean = mean;
                this.covariance = covariance;
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
    }

}
