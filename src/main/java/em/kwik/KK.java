package em.kwik;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * KlustaKwik computing engine.
 * 
 * Translation of KlustaKwik 1.7 release source to Java.
 * This after I became very tired of maintaining the nasty C++ code.
 * 
 * @author Peter N. Steinmetz
 * Created on Nov 18, 2006
 */
public class KK {
  private static Log log = LogFactory.getLog(KK.class);
  
  // :TODO: All this data should be divided into a model, the data, and temporaries

  boolean fullStep; // Indicates that the next E-step should be a full step (no time saving)

  int eStepCalls;

  int nPoints; // number of points in data

  int nDims;
  
  int nDimsSq;

  float[][] data;   // Input data indexed for point p, dimension d

  int maxClusters;  // maximum number of clusters considered and allocated for

  GaussianClusterModels clMods;
  
  double [][] pointScore;  // minus log likelihood scores less penalty, 
                           //  indexed first by point p then cluster c
                           // small scores are the best
  
  float[] dist; // working vector for distance calculation, ndims long

  float[] root; // working vector for result of Chol*Root = Vec

  int[] clust; // clust[p] = best cluster for point p in current iteration

  int[] oldClust; // oldClust[p] = cluster for point p on prior iteration

  int[] bestClass; // best class for each point achieved on any iteration

  int[] class2; // class2[p] = second best cluster for point p in current iteration

  int[] nClassMembers; // nClassMembers[maxClusters] = number of points in each class
  
  float[][] chol;     // pChol[c] is the nDim squared Cholesky matrix for cluster c

  double[] deletionLoss; // deletionLoss[maxClusters] = the increase in score by deleting the cluster

  KlustaSave kSave;

  KKOptions options;

  Random rng;

  /**
   * private constructor, clients use builder
   */
  private KK() {
    options = new KKOptions();
    fullStep = true;
    eStepCalls = 0;
    rng = new Random();
  }

  private void allocateArrays(final int maxPossibleClusters, final int nDimsA, final int nPointsA) {
    this.maxClusters = maxPossibleClusters;
    this.nPoints = nPointsA;
    this.nDims = nDimsA;
    this.nDimsSq = nDims * nDims;

    data = new float[nPoints][nDims];
    clMods = new GaussianClusterModels(nDims,maxPossibleClusters);

    pointScore = new double[nPoints][maxPossibleClusters];
    dist = new float[nDims];
    root = new float[nDims];
    
    clust = new int[nPoints];
    oldClust = new int[nPoints];
    class2 = new int[nPoints];
    nClassMembers = new int[maxClusters];

    bestClass = new int[nPoints];
    chol = new float[maxClusters][];
    for (int ai = 0; ai< chol.length; ai++) chol[ai] = new float[nDimsSq];
    deletionLoss = new double[maxClusters];
  }

  /**
   * Build new KK with random subset these data points
   */
  public KK buildWithRandomSubset(final double fracNewPoints) {
    if (fracNewPoints<0.0 || fracNewPoints > 1.0) {
      throw new IllegalArgumentException("fraction of new points must be between 0 and 1, inclusive");
    }
    KK res = new KK();
    // nClustersAlive set by reindex
    res.fullStep = this.fullStep;
    res.eStepCalls = 0;
    res.options = (KKOptions)this.options.clone();
    res.rng = this.rng;

    res.allocateArrays(this.maxClusters, this.nDims, 
                        (int) Math.floor(this.nPoints * fracNewPoints));
    int stepInOriginalData = this.nPoints/res.nPoints;
    for (int i = 0; i < res.nPoints; i++) {
      int pSrc = i * stepInOriginalData + rng.nextInt(stepInOriginalData);
      for (int di = 0; di < this.nDims; di++) {
        res.data[i*res.nDims+di] = this.data[pSrc*this.nDims+di];
      }
    }
    
    return res;
  }

  /**
   * Build from Features.
   * 
   * Data is normalized before being clustered.
   * 
   * @param feat Features to cluster
   * @param maxClusters maximum possible # of clusters to allocate space for
   * @param saveModel whether to construct a KlustaSave object for saving state information
   * 
   * @throws IOException 
   */
  public static KK buildFromFeatures(final Features feat, final int maxClusters,
      final boolean saveModel) throws IOException {
    
    KK res = new KK();
    res.allocateArrays(maxClusters, feat.getNumFeatures(), feat.getNumPoints());

    if (saveModel) {
      KlustaSave kSv = res.buildSizedSave();
      res.kSave = kSv;
      res.options.setSaveModel(true);
    }

    res.loadFeatures(feat);
    
    log.info("Loaded " + feat.getNumPoints() + 
                " data points of dimension " + feat.getNumFeatures());

    return res;
  }
  
  /**
   * load features into data and normalize
   */
  public void loadFeatures(final Features feat) {
    // load data into array
    for (int p = 0; p < feat.getNumPoints(); p++) {
      for (int i = 0; i < feat.getNumFeatures(); i++) {
        data[p][i] = feat.features[i][p];
      }
    }
    normalizeData();
  }
  
  /**
   * save cluster centers for later output
   */
  void saveBestMeans(KlustaSave kSv) {

    kSv.cEStepCallsSave = kSv.cEStepCallsLast;
    kSv.nDimsBest = nDims;
    kSv.nBestClustersAlive = clMods.getNumClustersAlive();

    GaussianClusterModels.LiveClusterIterator lci = clMods.new LiveClusterIterator();
    while (lci.hasNext()) {
      int c = lci.next();

      // save for writing of Cholesky, which has gaps
//      kSv.bestAliveIndex[cc] = c; // save without gaps:  cc, not c
//      kSv.bestWeight[cc] = clMods.weight[c];
//      for (int i = 0; i < nDims; i++) {
//        kSv.bestMean[cc * nDims + i] = clMods.mean[c * nDims + i]; // save without gaps:  cc, not c
//      }

      if (log.isDebugEnabled()) {
        if (c == 0) log.debug("mean (at SaveBestMeans):");
// :TODO: needs routine to write matrix        outputWriter.outputAsMatrix(mean, c * nDims, 1, nDims);
        log.debug("... best:  ");
//        outputWriter.outputAsMatrix(kSv.bestMean, cc * nDims, 1, nDims);
      }
    }

    // Save best Chol matrix.
    // Don't bother with cluster 0.
    for (int cc = 1; cc < kSv.nBestClustersAlive; cc++) {
      int c = kSv.bestAliveIndex[cc];

      for (int i = 0; i < kSv.nDimsBest; i++)
        for (int j = 0; j <= i; j++) {
          kSv.pBestChol[c][i * kSv.nDimsBest + j] = kSv.pChol[c][i
              * kSv.nDimsBest + j];
        }
    }
  }
  
  /**
   * retrieve associated KlustaSave
   */
  public KlustaSave getKlustaSave() {
    return kSave;
  }
  
  /**
   * save current class assignments as best
   */
  public void saveCurrentClassesAsBest() {
    for (int p = 0; p < nPoints; p++) {
      bestClass[p] = clust[p];
    }
  }

  /**
   * build a KlustaSave of size for this model and data
   */
  public KlustaSave buildSizedSave() {
    KlustaSave res = new KlustaSave(maxClusters, nDims);
    return res;
  }

  /**
   * set seed for random number generation
   */
  public void setRngSeed(final long seed) {
    rng.setSeed(seed);
  }
  
  /**
   * set whether to dump distances as calculating CEM
   */
  public void setDistanceDump(final boolean dumpDistances) {
    options.setDistDump(dumpDistances);
  }

  /**
   * set penalty mix
   */
  public void setPenaltyMix(final double mix) {
    options.setPenaltyMix(mix);
  }

  /**
   * Normalize data in arrays so is in range 0 to 1. 
   * 
   * This helps with large inputs and is required for proper 
   * comparison to cluster 0 with it's uniform distribution over
   * 0..1.
   */
  void normalizeData() {
    for (int i = 0; i < nDims; i++) {

      //calculate min and max
      float min = Float.POSITIVE_INFINITY;
      float max = Float.NEGATIVE_INFINITY;
      for (int pi = 0; pi < nPoints; pi++) {
        float val = data[pi][i];
        if (val > max) max = val;
        if (val < min) min = val;
      }

      // Save min and max value for each feature
      if (options.isSaveModel()) {
        kSave.dataMin[i] = min;
        kSave.dataMax[i] = max;
      }

      if (log.isDebugEnabled()) {
        log.debug("normalizeData[" + i + "]: min=" + min + ",max="
            + max);
      }

      // now normalize
      for (int p = 0; p < nPoints; p++) {
        data[p][i] = (data[p][i] - min) / (max - min);
      }
    }
  }

  /**
   * load cluster assignments
   */
  public void loadClusters(Clusters clusts) {
    if (clusts.getNumPoints() != nPoints) {
      throw new IllegalArgumentException("number of points must agree");
    }

    clMods.setFirstNAlive(clusts.getNumClusters()+1);

    for (int p = 0; p < nPoints; p++) {
      clust[p] = clusts.assignments[p];
    }
    
  }
  
  /**
   * assign random clusters for points
   */
  public void setRandomClusters(final int nStartingClusters) {
    if (nStartingClusters > 1) {
      for (int p = 0; p < nPoints; p++) {
        clust[p] = rng.nextInt(nStartingClusters - 1) + 1;
      }
    } else {
      Arrays.fill(clust, 0);
    }

    clMods.setFirstNAlive(Math.max(nStartingClusters, 1));
  }
  
  /** 
   * Compute the complexity penalty a number of clusters
   * bearing in mind that cluster 0 has no free params except p.
   * 
   * @param number of clusters
   */
  float penalty(int n) {
    int nParams;

    if (n == 1) return 0;

    nParams = (nDims * (nDims + 1) / 2 + nDims + 1) * (n - 1); // each has cov, mean, &p

    double aicPenalty = nParams * 2;

    double bicPenalty = nParams * Math.log(nPoints) / 2;

    // return mixture of AIC and BIC
    return (float) ((1.0 - options.getPenaltyMix()) * aicPenalty + options.getPenaltyMix() * bicPenalty);
  }

  /**
   * Compute total score for all points in currently assigned clusters.
   * 
   * Requires M, E, and C steps to have been run
   */
  float computeScore() {
    
    float score = penalty(clMods.getNumClustersAlive());
    for (int p = 0; p < nPoints; p++) {
      score += pointScore[p][clust[p]];
    }
    
    if (log.isDebugEnabled()) {
      GaussianClusterModels.LiveClusterIterator lci = clMods.new LiveClusterIterator();
      while (lci.hasNext()) {
        int c = lci.next();
        double tScore = 0;
        for (int p = 0; p < nPoints; p++)
          if (clust[p] == c) tScore += pointScore[p][clust[p]];
        log.debug("class " + c + " has subscore " + tScore);
      }
    }
    
    return score;
  }

  /**
   * M-step: Calculate mean, cov, and weight for each living class
   * also deletes any classes with less points than nDim
   */
  void mStep() {

    // clear arrays
    Arrays.fill(nClassMembers, 0);
    clMods.clearModels();

    // Accumulate total number of points in each class
    for (int p = 0; p < nPoints; p++)
      nClassMembers[clust[p]]++;

    // check for any dead classes
    boolean needsReindex = false;
    GaussianClusterModels.LiveClusterIterator lci = clMods.new LiveClusterIterator();
    while (lci.hasNext()) {
      int c = lci.next();
      if ((c > 0) && (nClassMembers[c] <= nDims)) {
        clMods.setClassAlive(c,false);
        needsReindex = true;
        log.info("Deleted class " + c + " : not enough members");
      }
    }
    if (needsReindex) clMods.reindex();
    
    // Normalize by total number of points to give class weight
    lci = clMods.new LiveClusterIterator();
    while (lci.hasNext()) {
      int c = lci.next();
      // add "noise point" to make sure weight for noise cluster never gets to zero
      if (c == 0) {
        clMods.weight[c] = (float)(nClassMembers[c] + options.getNoisePoint()) / (nPoints + options.getNoisePoint());
      } else {
        clMods.weight[c] = (float)(nClassMembers[c]) / (nPoints + options.getNoisePoint());
      }
    }
    
    // Accumulate sums for mean calculation
    for (int p = 0; p < nPoints; p++) {
      int c = clust[p];
      for (int i = 0; i < nDims; i++) {
        clMods.mean[c * nDims + i] += data[p][i];
      }
    }

    // divide by number of points in class
    lci = clMods.new LiveClusterIterator();
    while (lci.hasNext()) {
      int c = lci.next();
      if (nClassMembers[c] == 0) continue; // avoid divide by 0
      for (int i = 0; i < nDims; i++)
        clMods.mean[c * nDims + i] /= nClassMembers[c];
    }

    // Accumulate sums for covariance calculation
    for (int p = 0; p < nPoints; p++) {
      int c = clust[p];

      // calculate distance from mean
      for (int i = 0; i < nDims; i++) {
        dist[i] = data[p][i] - clMods.mean[c * nDims + i];
      }

      for (int i = 0; i < nDims; i++) {
        for (int j = i; j < nDims; j++) {
          clMods.cov[c * nDimsSq + i * nDims + j] += dist[i] * dist[j];
        }
      }
    }

    // divide by number of points in class - 1
    lci = clMods.new LiveClusterIterator();
    while (lci.hasNext()) {
      int c = lci.next();
      if (nClassMembers[c] <= 1) continue; // avoid divide by 0     
      for (int i = 0; i < nDims; i++) {
        for (int j = i; j < nDims; j++) {
          clMods.cov[c * nDimsSq + i * nDims + j] /= (nClassMembers[c] - 1);
        }
      }
    }

    if (log.isDebugEnabled()) {
      lci = clMods.new LiveClusterIterator();
      while (lci.hasNext()) {
        int cx = lci.next();
        log.debug("clust " + cx + " - weight " + clMods.weight[cx]);
        log.debug("mean: ");
//        outputWriter.outputAsMatrix(mean, cx * nDims, 1, nDims);
      }
    }
  }

  /**
   * E-step. Calculate score for each point as if it belonged to every living class.
   * Will delete a class if the covariance matrix is singular.
   */
  void eStep() {

    eStepCalls++;
    if (options.isSaveModel()) kSave.cEStepCallsLast = eStepCalls;
    int nSkipped = 0;

    // start with cluster 0 - uniform distribution over space 
    // because we have normalized all dims to 0...1, density will be 1 over this interval
    for (int p = 0; p < nPoints; p++) {
      pointScore[p][0] = -Math.log(clMods.weight[0]);
    }

    // constant term added into score
    double constScoreTerm = Math.log(2 * Math.PI) * nDims / 2;
    
    // loop over alive clusters other than 0
    GaussianClusterModels.LiveClusterIterator lci = clMods.new LiveClusterIterator();
    while (lci.hasNext()) {
      int c = lci.next();
      if (c==0) continue;

      // calculate cholesky decomposition of covariance matrix for class c
      if (!LinearAlgebra.cholesky(clMods.cov, c * nDimsSq, chol[c], nDims)) {
        // If Cholesky returns false, the matrix is not positive
        // definite, so kill the class.
        log.info("Deleting class " + c + ": covariance matrix is singular");
        clMods.setClassAlive(c, false);
        continue;
      }

      // logRootDet is given by log of product of diagonal elements,
      //  equivalently, the sum of the logs of diagonal elements
      double logRootDet = 0;
      for (int i = 0; i < nDims; i++)
        logRootDet += Math.log(chol[c][i * nDims + i]);
      
      double logWeight = Math.log(clMods.weight[c]);
      
      for (int p = 0; p < nPoints; p++) {
        // to save time -- only recalculate if score has changed by threshold,
        //  it is not a full step, and the point is in the same class as last time
        if (!fullStep && clust[p] == oldClust[p]
            && pointScore[p][c] - pointScore[p][clust[p]] > options.getDistThreshold()) {
          nSkipped++;
          continue;
        }

        // calculate data minus class mean
        for (int i = 0; i < nDims; i++) {
          dist[i] = data[p][i] - clMods.mean[c * nDims + i];
        }

        // calculate root vector which is solution to Chol*Root = dist
        LinearAlgebra.triSolve(chol[c], dist, root);
        
        // Compute Mahalanobis distance squared
        double mahal = 0;
        for (int i = 0; i < nDims; i++)
          mahal += root[i] * root[i];
        
        pointScore[p][c] = mahal / 2 + logRootDet - logWeight + constScoreTerm;

        if (log.isDebugEnabled()) {
          if (p == 0) {
            log.debug("cluster # " + c);
            log.debug("Cholesky");
            //outputWriter.outputAsMatrix(chol[c], 0, nDims, nDims);
            StringBuffer vecElemsBuf = new StringBuffer();
            for (int i=0; i< nDims; i++) {
              vecElemsBuf.append(root[i]).append(" ");
            }
            log.debug("root vector:" + vecElemsBuf.toString());
            log.debug("First point's score = " + mahal / 2 + " + "
                + logRootDet + " - " + logWeight + " + " + constScoreTerm + 
                " = " + pointScore[p][c]);
          }
        }

      } // over points, p

    } // over clusters, c

  }

  /**
   * Choose best and second best class for each point out of those living
   */
  void cStep() {

    for (int p = 0; p < nPoints; p++) {
      oldClust[p] = clust[p];
      double bestScore = Float.POSITIVE_INFINITY;
      double secondScore = Float.POSITIVE_INFINITY;
      int bestClassNum = 0;
      int secondClass = 0;

      GaussianClusterModels.LiveClusterIterator lci = clMods.new LiveClusterIterator();
      while (lci.hasNext()) {
        int c = lci.next();
        double curScore = pointScore[p][c];
        if (curScore < bestScore) {
          secondClass = bestClassNum;
          bestClassNum = c;
          secondScore = bestScore;
          bestScore = curScore;
        } else if (curScore < secondScore) {
          secondClass = c;
          secondScore = curScore;
        }
      }
      clust[p] = bestClassNum;
      class2[p] = secondClass;
    }
  }

  /**
   * Sometimes deleting a cluster will improve the score, when you take into
   * account the penalty. This function checks if this is the case. 
   * It deletes at most one cluster, that leading to the greatest 
   * improvement in score.
   * 
   * @return old cluster number which was deleted, or -1 if none deleted
   */
  int considerDeletion() {

    compDeletionLoss();

    // find cluster with least gain in score, excluding cluster 0
    int candidateClass = -1;
    double loss = Double.POSITIVE_INFINITY;
    for (int c = 1; c < maxClusters; c++) {
      if (deletionLoss[c] < loss) {
        loss = deletionLoss[c];
        candidateClass = c;
      }
    }

    // what is the loss in penalty when deleting one cluster?
    float deltaPen = penalty(clMods.getNumClustersAlive()) - 
                      penalty(clMods.getNumClustersAlive() - 1);

    // is it worth it?
    int clustDeleted = -1;
    if (loss < deltaPen) {
      log.info("Deleting clust " + candidateClass + ". Gain " + loss
          + " but lose " + deltaPen + " due to penalty.");
      clustDeleted = candidateClass;
      // set it to dead
      clMods.setClassAlive(candidateClass,false);

      // re-allocate all of its points
      for (int p = 0; p < nPoints; p++) {
        if (clust[p] == candidateClass) clust[p] = class2[p];
      }
      clMods.reindex();
    }
    return clustDeleted;
  }

  /**
   * Compute the loss which will occur for each cluster if all points 
   * in that cluster are deleted and re-assigned to their second best
   * cluster.
   */
  void compDeletionLoss() {
    for (int c = 0; c < maxClusters; c++) {
      if (clMods.isAlive(c))
        deletionLoss[c] = 0;
      else
        deletionLoss[c] = Float.POSITIVE_INFINITY; // don't delete classes that are already dead
    }

    // compute losses caused by deleting clusters and reallocating to second best cluster
    for (int p = 0; p < nPoints; p++) {
      deletionLoss[clust[p]] += pointScore[p][class2[p]]
          - pointScore[p][clust[p]];
    }
  }

  /**
   * For each cluster, try to split it in two. If that improves the score, do it.
   * @return true if split was successful
   */
  boolean trySplits() {
    boolean didSplit = false;

    if (clMods.getNumClustersAlive() >= maxClusters - 1) {
      log.info("Won't try splitting - already at maximum number of clusters");
      return false;
    }

    // set up kk3
    KK kk3 = new KK();
    kk3.allocateArrays(maxClusters,this.nDims,this.nPoints);
    kk3.options.setPenaltyMix(options.getPenaltyMix());
    System.arraycopy(data, 0, kk3.data, 0, data.length);

    float score = computeScore();

    // loop through clusters, trying to split, but not cluster 0
    GaussianClusterModels.LiveClusterIterator lci = clMods.new LiveClusterIterator();
    while (lci.hasNext()) {
     
      int c = lci.next();
      if (c==0) continue;

      // set up kk2 structure to contain points of this cluster only

      // count number of points and allocate memory
      int numPointsForCluster = 0;
      for (int p = 0; p < nPoints; p++)
        if (clust[p] == c) numPointsForCluster++;
      if (numPointsForCluster == 0) continue;
      KK kk2 = new KK();
      kk2.allocateArrays(maxClusters,this.nDims,numPointsForCluster);
      kk2.setPenaltyMix(options.getPenaltyMix());
      kk2.options.setNoisePoint(0);

      // put data into K2
      int p2 = 0;
      for (int p = 0; p < nPoints; p++) {
        if (clust[p] == c) {
          kk2.data[p2] = data[p];
        }
      }

      // find an unused cluster
      int unusedCluster = -1;
      for (int c2 = 1; c2 < maxClusters; c2++) {
        if (!clMods.isAlive(c2)) {
          unusedCluster = c2;
          break;
        }
      }
      if (unusedCluster == -1) {
        log.info("No free clusters, abandoning split");
        return false;
      }

      // do it
      log.info("Trying to split cluster " + c + "(" + kk2.nPoints
          + " points)");
      kk2.setRandomClusters(2);
      float unsplitScore = kk2.CEM(false);
      kk2.setRandomClusters(3); // (3 = 2 clusters + 1 unused noise cluster)
      float splitScore = kk2.CEM(false);

      if (kk2.clMods.getNumClustersAlive()<2) {
        log.info("Split failed, abandoning split");
        return false;
      }
      if (splitScore < unsplitScore) {

        // assign clusters to kk3
        for (int c2 = 0; c2 < maxClusters; c2++)
          kk3.clMods.setClassAlive(c2,false);
        p2 = 0;
        for (int p = 0; p < nPoints; p++) {
          if (clust[p] == c) {
            if (kk2.clust[p2] == 1)
              kk3.clust[p] = c;
            else if (kk2.clust[p2] == 2)
              kk3.clust[p] = unusedCluster;
            else
              throw new RuntimeException("split should only produce 2 clusters");
            p2++;
          } else
            kk3.clust[p] = clust[p];
          kk3.clMods.setClassAlive(kk3.clust[p],true);
        }
        kk3.clMods.reindex();

        // compute scores
        kk3.mStep();
        kk3.eStep();
        float newScore = kk3.computeScore();
        log.info("Splitting cluster " + c
            + " changes total score from " + score + " to " + newScore);

        if (newScore < score) {
          didSplit = true;
          log.info("So it's getting split into cluster "
              + unusedCluster);
          // so put clusters from K3 back into main KK struct (K1)
          for (int c2 = 0; c2 < maxClusters; c2++)
            clMods.setClassAlive(c2,kk3.clMods.isAlive(c2));
          for (int p = 0; p < nPoints; p++)
            clust[p] = kk3.clust[p];
        } else {
          log.info("So it's not getting split.");
        }
      }
    }
    return didSplit;
  }
  
  /**
   * Perform a whole CEM algorithm.
   * 
   * Assumes some initial cluster assignment has been made, random or from a Clusters object.
   * 
   * @param recurse if should recursively attempt splits
   */
  float CEM(boolean recurse) {
    boolean didSplit;

    // main loop
    int iter = 0;
    fullStep = true;
    float score = 0;
    float bestScore = Float.POSITIVE_INFINITY;
    int nChanged = 0;
    boolean lastStepFull = false;
    do {

      // M-step - calculate class weights, means, and covariance matrices for each class
      mStep();

      // E-step - calculate scores for each point to belong to each class
      eStep();

      // dump distances if required
//      if (distDump) outputWriter.outputAsMatrix(pointScore, 0, 1, maxClusters);

      // C-step - choose best class for each 
      cStep();

      // Would deleting any classes improve things?
      if (recurse) considerDeletion();

      // Calculate number of points changed by M E C steps and deletions
      nChanged = 0;
      for (int p = 0; p < nPoints; p++) {
        nChanged += (oldClust[p] != clust[p] ? 1 : 0);
      }

      score = computeScore();
      
      // save cluster centers for later output, but
      // only if not just horsing around with testing
      // splits.
      if (options.isSaveModel() && recurse && (score < bestScore)) {
        saveBestMeans(kSave);
        // this might be better than the score
        // returned at the end of the iteration
        kSave.BestScoreSave = score;
      }

      if (score < bestScore) bestScore = score;

      if (log.isInfoEnabled()) {
        log.info("Iteration " + iter + (fullStep ? 'F' : 'Q')
            + ": clusters " + clMods.getNumClustersAlive() + " Score " + 
            score + " nChanged "
            + nChanged + " tag:" +
            (options.isSaveModel()? Integer.toString(kSave.cEStepCallsLast) : null) );
      }

      iter++;

      // Next step a full step?
      lastStepFull = fullStep;
      fullStep = (nChanged > options.getFracPointsChangedThreshold() * nPoints
          || nChanged == 0 || (iter % options.getFullStepEvery()) == 0);
      if (iter > options.getMaxIter()) {
        log.info("Maximum iterations exceeded");
        break;
      }

      // try splitting
      if (recurse
          && options.getSplitEvery() > 0
          && (iter % options.getSplitEvery() == options.getSplitEvery() - 1 || (nChanged == 0 && lastStepFull))) {
        didSplit = trySplits();
      } else
        didSplit = false;

    } while (nChanged > 0 || !lastStepFull || didSplit);

    return score;
  }
  
  /**
   * Perform clustering, using a random subset of data.
   */
  public float clusterWithSubset(final double subFraction, final int nStartingClusters) {
    KK kkSub = buildWithRandomSubset(subFraction);
    log.info("--- Clustering on subset of " + kkSub.nPoints + " points.");
    kkSub.setRandomClusters(nStartingClusters);
    kkSub.CEM(true);
    this.clMods.copyFrom(kkSub.clMods);
    log.info("--- Evaluating fit on full set of " + nPoints + " points.");
    eStep();
    cStep();
    
    return computeScore();
  }

}
